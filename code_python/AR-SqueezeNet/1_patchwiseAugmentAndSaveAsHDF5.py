from csv import excel
from re import search
import SimpleITK as sitk 
from pathlib import Path
import sys 
sys.path.append(fr"sourcecode/classification_lesion/Code_general")
from dataUtil import DataUtil 
from medImageProcessingUtil import MedImageProcessingUtil
import tables
import os 
import numpy as np
from glob import glob 
import matplotlib.pyplot as plt 
from augmentation3DUtil import Augmentation3DUtil
from augmentation3DUtil import Transforms
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.morphology import opening, closing,binary_dilation
from skimage.morphology import disk
from skimage.measure import regionprops
from skimage.measure import label as ConnectedComponent
from skimage.transform import resize as skresize
import pandas as pd 
from tqdm import tqdm

SOURCECODE_DIR = os.getcwd() + '/sourcecode/classification_lesion'
# LESION_DIR = 'px1Label_HO'
# RAD_NAME = LESION_DIR.split('_')[-1]
RAD_NAMES = ['JR', 'LKB', 'RDW', 'SV', 'SHT']
AUGMENTATION = [4, 8]


def _getAugmentedData(imgs,masks,nosamples):
    
    """ 
    This function defines different augmentations/transofrmation sepcified for a single image 
    img,mask : to be provided SimpleITK images 
    nosamples : (int) number of augmented samples to be returned
    
    """
    au = Augmentation3DUtil(imgs,masks=masks)

    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.05,0.05))
    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.02,0.05))
    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.01,0.05))
    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.03,0.05))

    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = 1)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = -1)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = 2)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = -2)
    au.add(Transforms.FLIPHORIZONTAL,probability = 0.5)

    imgs, augs = au.process(nosamples)

    return imgs,augs

def getBoundingBoxResampledImage(img,ls,newsize2D):

    ''' 
    Crops the volume to the bounding box of the prostate and resizes the volume to the given newsize2D 
    
    img : sitk Image of sequence 
    pm : prostate\capsule mask of the image 
    ls : lesion mask of the image
    newsize2D : dimensions required for x and y directions. Dimensions of z will be retained after cropping
    '''
    spacing = img.GetSpacing()
    size = img.GetSize()

    zsize = size[2]
 
    rimg = DataUtil.resampleimagebysize(img,(newsize2D,newsize2D,zsize))
    rls = DataUtil.resampleimagebysize(ls,(newsize2D,newsize2D,zsize),interpolator=sitk.sitkNearestNeighbor)

    rls = DataUtil.convert2binary(rls)

    return (rimg,rls)


def createHDF5(hdf5path,splitsdict,patchSize,depth):
    
    """
    splitspathname : name of the file (json) which has train test splits info 
    patchSize : x,y dimension of the image 
    depth : z dimension of the image 
    """
    # outputfolder = fr"{SOURCECODE_DIR}/outputs/hdf5/hdf5_{RAD_NAME}_{AUGMENTATION[0]}{AUGMENTATION[1]}/{hdf5path}"
    outputfolder = fr"{SOURCECODE_DIR}/outputs/hdf5/{hdf5path}"
    Path(outputfolder).mkdir(parents=True, exist_ok=True)

    img_dtype = tables.Float32Atom()

    if depth > 1:
        data_shape = (0, depth,3, patchSize[0], patchSize[1])
        data_chuck_shape = (1, depth,3,patchSize[0],patchSize[1])

    else:
        data_shape = (0, 3, patchSize[0], patchSize[1])
        data_chuck_shape = (1,3,patchSize[0],patchSize[1])


    filters = tables.Filters(complevel=5)

    splitsdict = DataUtil.readJson(splitspath)

    phases = np.unique(list(splitsdict.values()))

    for phase in phases:
        hdf5_path = fr'{outputfolder}/{phase}.h5'

        if os.path.exists(hdf5_path):
            Path(hdf5_path).unlink()

        hdf5_file = tables.open_file(hdf5_path, mode='w')


        data = hdf5_file.create_earray(hdf5_file.root, "data", img_dtype,
                                            shape=data_shape,
                                            chunkshape = data_chuck_shape,
                                            filters = filters)


        hdf5_file.close()


def _addToHDF5(sample,phase,splitspathname):
    
    """
    imgarr : input image sample (ex 2 slices of the image)
    maskarr : output mask (ex. lesion segmentation mask)
    phase : phase of that image (train,test,val)
    splitspathname : name of the file (json) which has train test splits info 
    """
    # outputfolder = fr"outputs/hdf5/hdf5_{RAD_NAME}_{AUGMENTATION[0]}{AUGMENTATION[1]}/{splitspathname}"
    outputfolder = fr"outputs/hdf5/hdf5_{RAD_NAME}_newsplits"

    hdf5_file = tables.open_file(fr'{SOURCECODE_DIR}/{outputfolder}/{phase}.h5', mode='a')

    data = hdf5_file.root["data"]

    data.append(sample[None])
    
    hdf5_file.close()

def getAugmentedData(folderpath, lsfile, lesion, nosamples = None):
    
    """
    folderpath : path to folder containing images, mask
    nosamples : Number of augmentations to be performed
    """
    folderpath = Path(folderpath)

    # try:
    #     ext = folderpath.glob(fr"T2W_std*").__next__().suffix
    # except:
    #     import pdb 
    #     pdb.set_trace()

    # if ext == ".gz":
    #     ext = ".".join(glob(fr"{folderpath}/**")[0].split("/")[-1].split(".")[-2:])
    # else:
    #     ext = ext.replace(".","")

    ext = 'nii.gz'
    t2w = sitk.ReadImage(str(folderpath.joinpath(fr"T2W_std.{ext}")))
    adc = sitk.ReadImage(str(folderpath.joinpath(fr"ADC_reg.{ext}")))

    imgs = [t2w,adc] 

    pm = sitk.ReadImage(str(folderpath.joinpath(fr"PM.{ext}")))
    pm = DataUtil.convert2binary(pm)

    ls = sitk.ReadImage(str(lsfile))
    ls = DataUtil.convert2binary(ls)

    masks = [pm,ls]

    ret = []
    
    orgimg,augs = _getAugmentedData(imgs,masks,nosamples)
    ret.append((orgimg))

    if augs is not None:
        for i in range(len(augs)):
            ret.append(augs[i])

    return ret

def normalizeImage(img,_min,_max,clipValue=None):

    imgarr = sitk.GetArrayFromImage(img)

    if clipValue is not None:
        imgarr[imgarr > clipValue] = clipValue 

    imgarr[imgarr < _min] = _min
    imgarr[imgarr > _max] = _max

    imgarr = (imgarr - _min)/(_max - _min)

    imgarr = imgarr.astype(np.float32)

    return imgarr


def addToHDF5(t2w,adc,pm,ls,phase,splitspathname,patchSize,t2wmin,t2wmax,adcmin,adcmax,name,label,dilate=None):
    
    """ 
    Collect samples from the cropped volume and add them into HDF5 file 

    img : SimpleITK image to be pre-processed
    pm : prostate mask as SimpleITK image to be preprocesed
    resample_spacing : spacing to which the image have to be resampled 
    crop_size : x,y,z crop size in mm 
    intensitystandardize : as a tuple (template image as SimpleITK image, Number of histogram levels, Number of match points)
    """
    pm = sitk.BinaryDilate(pm,2,sitk.sitkBall)

    names = [] 
    labels = []

    t2warr = normalizeImage(t2w,t2wmin,t2wmax,clipValue=t2wmax)
    adcarr = normalizeImage(adc,adcmin,adcmax,clipValue=3000)

    lsarr = sitk.GetArrayFromImage(ls)

    slnos = np.unique(np.nonzero(lsarr)[0])

    pmarr = sitk.GetArrayFromImage(pm)

    samples = None 

    spacing = t2w.GetSpacing()

    if dilate is not None:
        dilate = int(dilate/spacing[0])

    for i in slnos:
        slc = lsarr[i]
        pmslc = pmarr[i]

        labelarr = ConnectedComponent(slc)
        # labels = np.unique(labelarr[np.nonzero(labelarr)])

        # for propno in labels:

        if dilate is not None:
            peri = binary_dilation(slc,disk(dilate))
        else:
            peri = slc 

        peri = peri*pmslc
        peri = peri.astype(np.uint8)

        props = regionprops(peri)



        if not peri.sum() == 0:

            sample = np.zeros((3,patchSize[1],patchSize[0]))
            starty, startx, endy, endx = props[0].bbox 
            t2wsample = t2warr[i, starty:endy, startx:endx]
            adcsample = adcarr[i, starty:endy, startx:endx]
            mask  = lsarr[i, starty:endy, startx:endx]
            orgmask = slc[starty:endy, startx:endx]

            sample[0] = skresize(t2wsample,(patchSize[1],patchSize[0]),order=1)
            sample[1] = skresize(adcsample,(patchSize[1],patchSize[0]),order=1)
            sample[2] = sitk.GetArrayFromImage(DataUtil.resampleimagebysize(sitk.GetImageFromArray(mask),(patchSize[1],patchSize[0]),sitk.sitkNearestNeighbor))

            # samples = sample[None] if samples is None else np.vstack((samples,sample[None]))
            
            names.append(fr"{name}_{i}") # i  slice no 
            # labels.append(label)
            labels.append(label)
            _addToHDF5(sample,phase,splitspathname) #(patchwise)
            
    # _samples = np.zeros((depth,3,patchSize[1],patchSize[0]))
    # _samples[:samples.shape[0]] = samples

    # _addToHDF5(_samples,phase,splitspathname)
    # if label == 0:
        
    
    #     import pdb; pdb.set_trace()
    # return [name],[label]
    return names, labels #(patchwise)


def _getminmax(templatefolder,modality):
    img = sitk.ReadImage(fr"{templatefolder}/{modality}.nii")
    pm = sitk.ReadImage(fr"{templatefolder}/PM.nii")

    pm = DataUtil.convert2binary(pm)

    imgarr = sitk.GetArrayFromImage(img)
    pmarr = sitk.GetArrayFromImage(pm)

    maskedarr = imgarr*pmarr
    _min = maskedarr.min()
    _max = maskedarr.max()   

    return _min,_max  


if __name__ == "__main__":

    labelsdf = pd.read_csv(fr"{SOURCECODE_DIR}/outputs/labels/significance_prostatex.csv").set_index("Lesion")
    labelsdict = labelsdf.loc[labelsdf['Dataset'] == 'ProstateX1'].to_dict()

    cvsplits = 3
    dilate = None 

    for RAD_NAME in RAD_NAMES:
        print(RAD_NAME)

        splitspathname = fr"prostatex_1"
        # splitspathname = fr"new_splits"

        inputfoldername = fr"1_Original_Organized"
        
        
        newsize2D = (224,224) 
        depth = 0
        
        inpaint = False 

        splitspath = fr"{SOURCECODE_DIR}/outputs/splits/{splitspathname}.json"
        splitsdict = DataUtil.readJson(splitspath)

        cases = list(splitsdict.keys())

        hdf5path = fr"hdf5_{RAD_NAME}_48"
        # hdf5path = fr"hdf5_{RAD_NAME}_newsplits"

        createHDF5(hdf5path,splitsdict,newsize2D,depth)
        
        casenames = {} 
        casenames["train"] = [] 
        casenames["val"] = []
        casenames["test"] = [] 

        caselabels = {} 
        caselabels["train"] = [] 
        caselabels["val"] = [] 
        caselabels["test"] = [] 

        templateimg = sitk.ReadImage(fr"{SOURCECODE_DIR}/Template/T2W.nii.gz")
        templatepm = sitk.ReadImage(fr"{SOURCECODE_DIR}/Template/PM.nii.gz")
        templatepm = DataUtil.convert2binary(templatepm)
        masked = sitk.Mask(templateimg,templatepm)
        maskedarr = sitk.GetArrayFromImage(masked)
        
        t2wmin = np.min(maskedarr)
        t2wmax = np.max(maskedarr)

        adcmin = 0 
        adcmax = 3000
        
        # Patient loop
        for j, name in enumerate(cases):
            dataset,pat = name.split("_")
            # dataset,pat,lsnum = name.split("_")
            patname = fr"{dataset}_{pat}"
            sb = Path(fr"annotations_master/Original_Sequences/{patname}")
            
            lesionpath = Path(fr"annotations_master/px1Label_{RAD_NAME}/{patname}/")
            # lesionpath = Path(fr"annotations_master/Original_Sequences/{name}")

            # getting the label 
            lsfiles = lesionpath.glob(fr"LS*.nii")
            # lsfiles = lesionpath.glob(fr"LS*.nii.gz")
            
            try:
                keys = [key for (key, value) in labelsdict['Sig'].items() if search(name, key)]
            except:
                import pdb; pdb.set_trace()
            
            # Lesion loop
            for k, lsfile in enumerate(lsfiles):
                lsfile = str(lsfile)
                lesion = lsfile.split('LS')[-1].split('.')[0]
                
                print(fr"Name: {patname}, Lesion: LS{lesion}, Progress: {j/len(cases)}")
                
                # label = 1 if j%2==0 else 0 
                label = labelsdict['Sig'][fr"{patname}_L{lesion}"]

                nosamples = AUGMENTATION[0] if label == 1 else AUGMENTATION[1]
                
                phase = splitsdict[name]

                if phase == "train":
                    ret = getAugmentedData(sb,lsfile, lesion,nosamples=nosamples)
                else:
                    ret = getAugmentedData(sb,lsfile, lesion,nosamples=None)

                for k,aug in enumerate(ret):
                
                    augt2w = aug[0][0]
                    augadc = aug[0][1]
                    augpm = aug[1][0]
                    augls = aug[1][1]
                    
                    casename = fr"{name}_L{lesion}" if k == 0 else fr"{patname}_L{lesion}_A{k}"

                    names,labels = addToHDF5(augt2w,augadc,augpm,augls,phase,hdf5path,newsize2D,t2wmin,t2wmax,adcmin,adcmax,casename,label,dilate=dilate)

                    casenames[phase].extend(names)
                    caselabels[phase].extend(labels)

        # outputfolder = fr"outputs/hdf5/hdf5_{RAD_NAME}_{AUGMENTATION[0]}{AUGMENTATION[1]}/{hdf5path}"
        outputfolder = fr"outputs/hdf5/{hdf5path}"

        for phase in ["train","val","test"]:
            hdf5_file = tables.open_file(fr'{SOURCECODE_DIR}/{outputfolder}/{phase}.h5', mode='a')
            hdf5_file.create_array(hdf5_file.root, fr'names', casenames[phase])
            hdf5_file.create_array(hdf5_file.root, fr'labels', caselabels[phase])

            hdf5_file.close()