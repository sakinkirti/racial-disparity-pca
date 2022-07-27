from csv import excel
from re import search
import SimpleITK as sitk 
from pathlib import Path
import sys 
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

# define some global variables
SOURCECODE_DIR = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa'
RAD_NAMES = ['ST', 'LB']
AUGMENTATION = [14, 2]

# some variables used in the script
labels_csv = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-AA/racial_disparity_ggg_filtered.csv'

""" 
This function defines different augmentations/transofrmation sepcified for a single image 
img,mask : to be provided SimpleITK images 
nosamples : (int) number of augmented samples to be returned
"""
def generateAugmentedData(imgs,masks,nosamples):
    # instantiate the augmenter
    au = Augmentation3DUtil(imgs,masks=masks)

    # generate a few different forms of the image and add to the augmenter
    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.05,0.05))
    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.02,0.05))
    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.01,0.05))
    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.03,0.05))
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = 1)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = -1)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = 2)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = -2)
    au.add(Transforms.FLIPHORIZONTAL,probability = 0.5)

    # process and save the images and augmentations
    imgs, augs = au.process(nosamples)
    return imgs,augs

''' 
Crops the volume to the bounding box of the prostate and resizes the volume to the given newsize2D 

img : sitk Image of sequence 
pm : prostate\capsule mask of the image 
ls : lesion mask of the image
newsize2D : dimensions required for x and y directions. Dimensions of z will be retained after cropping
'''
def getBoundingBoxResampledImage(img,ls,newsize2D):
    # get the image size and isolate z
    size = img.GetSize()
    zsize = size[2]
 
    # resample the image according to the lesion
    rimg = DataUtil.resampleimagebysize(img,(newsize2D,newsize2D,zsize))
    rls = DataUtil.resampleimagebysize(ls,(newsize2D,newsize2D,zsize),interpolator=sitk.sitkNearestNeighbor)
    rls = DataUtil.convert2binary(rls)

    # return a tuple containing the resampled images
    return (rimg,rls)

"""
splitspathname : name of the file (json) which has train test splits info 
patchSize : x,y dimension of the image 
depth : z dimension of the image 
"""
def createHDF5(hdf5path,splitsdict,patchSize,depth):
    # define the location to save the output path
    outputfolder = fr"/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-AA/{hdf5path}"
    Path(outputfolder).mkdir(parents=True, exist_ok=True)

    # define the image type to save to the hdf5 file
    img_dtype = tables.Float32Atom()

    # determine how many folder levels are required and generate as such
    if depth > 1:
        data_shape = (0, depth,3, patchSize[0], patchSize[1])
        data_chuck_shape = (1, depth,3,patchSize[0],patchSize[1])
    else:
        data_shape = (0, 3, patchSize[0], patchSize[1])
        data_chuck_shape = (1,3,patchSize[0],patchSize[1])

    # set the filters
    filters = tables.Filters(complevel=5)

    # read the splits and separate by train, val, test
    splitspath = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-AA/rdp-splits-json/racial-disparity-splits.json'
    splitsdict = DataUtil.readJson(splitspath)
    phases = np.unique(list(splitsdict.values()))

    # create each of the hdf5 files (one per train, val, test)
    for phase in phases:
        hdf5_path = fr'{outputfolder}/{phase}.h5'
        if os.path.exists(hdf5_path):
            Path(hdf5_path).unlink()
        hdf5_file = tables.open_file(hdf5_path, mode='w')
        data = hdf5_file.create_earray(hdf5_file.root, "data", img_dtype,
                                            shape=data_shape,
                                            chunkshape = data_chuck_shape,
                                            filters = filters)

        # close the hdf5 file
        hdf5_file.close()

"""
imgarr : input image sample (ex 2 slices of the image)
maskarr : output mask (ex. lesion segmentation mask)
phase : phase of that image (train,test,val)
splitspathname : name of the file (json) which has train test splits info 
"""
def _addToHDF5(sample,phase,splitspathname,outpath):
    # set the output folder and open the necessary hdf5 file
    outputfolder = outpath
    hdf5_file = tables.open_file(fr'{outputfolder}/{phase}.h5', mode='a')

    # add the image to the hdf5 file and close
    data = hdf5_file.root["data"]
    data.append(sample[None])
    hdf5_file.close()

"""
folderpath : path to folder containing images, mask
nosamples : Number of augmentations to be performed
"""
def getAugmentedData(folderpath, lsfile, lesion, nosamples = None):
    # read the T2W and ADC image
    folderpath = Path(folderpath)
    t2w = sitk.ReadImage(f'{str(folderpath)}/T2W_std.nii.gz')
    adc = sitk.ReadImage(f'{str(folderpath)}/ADC_reg.nii.gz')
    imgs = [t2w,adc] 

    # read the prostate and lesion masks
    pm = sitk.ReadImage(str(folderpath.joinpath(fr"PM.nii.gz")))
    pm = DataUtil.convert2binary(pm)
    ls = sitk.ReadImage(str(lsfile))
    ls = DataUtil.convert2binary(ls)
    masks = [pm,ls]

    # generate the augmented data
    ret = []
    orgimg,augs = generateAugmentedData(imgs,masks,nosamples)
    ret.append((orgimg))

    # add the augmented images to the return list and return
    if augs is not None:
        for i in range(len(augs)):
            ret.append(augs[i])
    return ret

'''
method to normal each method within a specific range
'''
def normalizeImage(img,_min,_max,clipValue=None):
    # store the image as an array
    imgarr = sitk.GetArrayFromImage(img)

    # clip the image if needed
    if clipValue is not None:
        imgarr[imgarr > clipValue] = clipValue 

    # normalize the image
    imgarr[imgarr < _min] = _min
    imgarr[imgarr > _max] = _max
    imgarr = (imgarr - _min)/(_max - _min)
    imgarr = imgarr.astype(np.float32)

    # return the image array
    return imgarr

""" 
Collect samples from the cropped volume and add them into HDF5 file 

img : SimpleITK image to be pre-processed
pm : prostate mask as SimpleITK image to be preprocesed
resample_spacing : spacing to which the image have to be resampled 
crop_size : x,y,z crop size in mm 
intensitystandardize : as a tuple (template image as SimpleITK image, Number of histogram levels, Number of match points)
"""
def addToHDF5(t2w,adc,pm,ls,phase,splitspathname,patchSize,t2wmin,t2wmax,adcmin,adcmax,name,label,outputpathname,dilate=None):
    # dilate the prostate mask and get an array of the pm
    pm = sitk.BinaryDilate(pm,2,sitk.sitkBall)
    pmarr = sitk.GetArrayFromImage(pm)

    # initialize the names and labels list
    names = [] 
    labels = []

    # normalize the t2w and adc images
    t2warr = normalizeImage(t2w,t2wmin,t2wmax,clipValue=t2wmax)
    adcarr = normalizeImage(adc,adcmin,adcmax,clipValue=3000)

    # get array of lesion
    lsarr = sitk.GetArrayFromImage(ls)
    slnos = np.unique(np.nonzero(lsarr)[0])

    spacing = t2w.GetSpacing()

    # set dilate if needed
    if dilate is not None:
        dilate = int(dilate/spacing[0])

    # iterate through the layers that have mask
    for i in slnos:
        slc = lsarr[i]
        pmslc = pmarr[i]

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
            
            # append the name and lablel to their respective list
            names.append(fr"{name}_{i}")
            labels.append(label)
            _addToHDF5(sample,phase,splitspathname,outputpathname)
            
    # return the names and labels
    return names, labels

'''
method not used
'''
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

'''
runs program
'''
def main():
    # read in the csv with the GGG scores (labels)
    labelsdf = pd.read_csv(labels_csv)

    # set dilate to none
    dilate = None 

    for RAD in RAD_NAMES:
        print(RAD)

        # set the variables used
        inputfoldername = fr'/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/AA_ggg-confirmed'
        newsize2D = (224,224) 
        depth = 0
        splitspath = fr'/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-AA/rdp-splits-json/racial-disparity-splits.json'

        # read in the templates
        templateimg = sitk.ReadImage(fr"/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/standardization/T2W_template.nii")
        templatepm = sitk.ReadImage(fr"/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/standardization/PM_T2W_template.nii")
        templatepm = DataUtil.convert2binary(templatepm)
        
        # read in the splits and save as test, val, or train
        splitsdict = DataUtil.readJson(splitspath)
        cases = list(splitsdict.keys())

        # set the folder name to store the hdf5 files and generate
        hdf5path = fr"rdp-hdf5/hdf5-{RAD}"
        createHDF5(hdf5path,splitsdict,newsize2D,depth)
        
        # create a dictionary for the cases and labels
        casenames = {} 
        casenames["train"] = [] 
        casenames["val"] = []
        casenames["test"] = [] 
        caselabels = {} 
        caselabels["train"] = [] 
        caselabels["val"] = [] 
        caselabels["test"] = [] 

        # mask the pm over the t2w template and get an array
        masked = sitk.Mask(templateimg,templatepm)
        maskedarr = sitk.GetArrayFromImage(masked)
        
        # get min and max maskedarr and set the min and max of adc
        t2wmin = np.min(maskedarr)
        t2wmax = np.max(maskedarr)
        adcmin = 0 
        adcmax = 3000
        
        # Patient loop
        for j, name in enumerate(cases):
            dataset,pat = name.split("-")
            patname = fr"{dataset}-{pat}"
            sb = Path(fr"{inputfoldername}/{patname}")
            
            # patient path
            lesionpath = Path(fr"/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/AA_lesion-masks/{RAD}/{patname}/")
            lsfiles = lesionpath.glob(fr"LS1*.nii.gz")
            
            # Lesion loop
            outputfolder = fr"/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-AA/{hdf5path}"
            for k, lsfile in enumerate(lsfiles):
                lsfile = str(lsfile)
                lesion = lsfile.split('LS')[-1].split('.')[0]
                print(fr"Name: {patname}, Lesion: LS{lesion}, Progress: {j/len(cases)}")
                
                # set the severity according to GGG score (binary classification)
                label = 1 if (labelsdf[labelsdf['ID'] == f'prostate_id {pat}']['GGG (1-5)'] > 1.0).bool() else 0

                nosamples = AUGMENTATION[1] if label == 1 else AUGMENTATION[0]
                
                phase = splitsdict[name]

                if phase == "train":
                    ret = getAugmentedData(sb, lsfile, lesion, nosamples=nosamples)
                elif phase == 'val':
                    nosamples = int(nosamples/2) if label == 1 else int(nosamples/2) + 1
                    ret = getAugmentedData(sb, lsfile, lesion, nosamples=nosamples)
                else:
                    ret = getAugmentedData(sb, lsfile, lesion, nosamples=4)

                for k,aug in enumerate(ret):
                
                    augt2w = aug[0][0]
                    augadc = aug[0][1]
                    augpm = aug[1][0]
                    augls = aug[1][1]
                    
                    casename = fr"{name}_L{lesion}" if k == 0 else fr"{patname}_L{lesion}_A{k}"

                    names,labels = addToHDF5(augt2w,augadc,augpm,augls,phase,hdf5path,newsize2D,t2wmin,t2wmax,adcmin,adcmax,casename,label,outputfolder,dilate=dilate)

                    casenames[phase].extend(names)
                    caselabels[phase].extend(labels)

        for phase in ["train","val","test"]:
            hdf5_file = tables.open_file(fr'{outputfolder}/{phase}.h5', mode='a')
            hdf5_file.create_array(hdf5_file.root, fr'names', casenames[phase])
            hdf5_file.create_array(hdf5_file.root, fr'labels', caselabels[phase])

            hdf5_file.close()

if __name__ == "__main__":
    main()