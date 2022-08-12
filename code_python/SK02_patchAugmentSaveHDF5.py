import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import tables
import glob
import torch
import h5py

from pathlib import Path
from skimage.morphology import binary_dilation, disk
from skimage.measure import regionprops
from skimage.transform import resize as skresize

from augmentation3DUtil import Augmentation3DUtil
from augmentation3DUtil import Transforms
from dataUtil import DataUtil

from torch.utils.data import Dataset

'''
@author Sakin Kirti
@date 5/20/2022

script to augment and save t2w, adc, ls files to HDF5 file training network

THIS FILE IS RUN ON A PERSONAL MACHINE
'''

# path variables
race = 'CA'

source_dir = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa'
labels_csv = f'{source_dir}/model-outputs-{race}/racial_disparity_ggg_filtered.csv'
splitspath = f'{source_dir}/model-outputs-{race}/rdp-splits-json/racial-disparity-splits.json'
t2w_template = f'/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/standardization/T2W_template.nii'
pm_template = f'/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/standardization/PM_T2W_template.nii'

rads = ['LB']
num_augs = [20, 10]

# ----- METHODS FOR AUGMENTATION OF DATA ----- #
def getAugmentedData(folderpath, lsfile, nosamples=None):
    '''
    folderpath : path to folder containing images, mask
    nosamples : Number of augmentations to be performed
    '''
    
    # read the T2W and ADC image
    folderpath = Path(folderpath)
    # all T2W images are standardized, but some are names T2W and others are names T2W_std, so set the image name
    t2w = sitk.ReadImage(f'{str(folderpath)}/T2W_std.nii.gz') 
    # all ADC images are registered, but some are named ADC and others are named ADC_reg, so set the image name
    adc = sitk.ReadImage(f'{str(folderpath)}/ADC_reg.nii.gz')
    imgs = [t2w,adc] 

    # read the prostate and lesion masks
    pm = sitk.ReadImage(f'{str(folderpath)}/PM.nii.gz')
    pm = DataUtil.convert2binary(pm)
    ls = sitk.ReadImage(str(lsfile))
    ls = DataUtil.convert2binary(ls)
    masks = [pm,ls]

    # generate the augmented data
    ret = []
    orgimg, augs = generateAugmentedData(imgs, masks, nosamples)
    ret.append((orgimg))

    # add the augmented images to the return list and return
    if augs is not None:
        for i in range(len(augs)):
            ret.append(augs[i])
    return ret

def generateAugmentedData(imgs, masks, nosamples):
    '''
    This function defines different augmentations/transofrmation sepcified for a single image 
    img,mask : to be provided SimpleITK images 
    nosamples : (int) number of augmented samples to be returned
    '''

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

def getBoundingBoxResampledImage(img, ls, newsize2D):
    ''' 
    Crops the volume to the bounding box of the prostate and resizes the volume to the given newsize2D 

    img : sitk Image of sequence 
    pm : prostate\capsule mask of the image 
    ls : lesion mask of the image
    newsize2D : dimensions required for x and y directions. Dimensions of z will be retained after cropping
    '''

    # get the image size and isolate z
    size = img.GetSize()
    zsize = size[2]
 
    # resample the image according to the lesion
    rimg = DataUtil.resampleimagebysize(img, (newsize2D, newsize2D, zsize))
    rls = DataUtil.resampleimagebysize(ls, (newsize2D, newsize2D, zsize), interpolator=sitk.sitkNearestNeighbor)
    rls = DataUtil.convert2binary(rls)

    # return a tuple containing the resampled images
    return (rimg, rls)

# ----- HDF5 METHODS ----- # 
def createHDF5(hdf5path, splitsdict, patchSize, depth):
    '''
    splitspathname : name of the file (json) which has train test splits info 
    patchSize : x,y dimension of the image 
    depth : z dimension of the image 
    '''

    # define the location to save the output path
    outputfolder = f'{source_dir}/model-outputs-{race}/{hdf5path}'
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

    # separate the splits by train, val, test
    phases = np.unique(list(splitsdict.values()))

    # create each of the hdf5 files (one per train, val, test)
    for phase in phases:
        hdf5_path = f'{outputfolder}/{phase}.h5'
        if os.path.exists(hdf5_path):
            Path(hdf5_path).unlink()
        hdf5_file = tables.open_file(hdf5_path, mode='w')
        data = hdf5_file.create_earray(hdf5_file.root, 'data', img_dtype,
                                            shape=data_shape,
                                            chunkshape = data_chuck_shape,
                                            filters = filters)

        # close the hdf5 file
        hdf5_file.close()

def addToHDF5(t2w, adc, pm, ls, phase, splitspathname, patchSize, t2wmin, t2wmax, adcmin, adcmax, name, label, outputpathname, dilate=None):
    '''
    Collect samples from the cropped volume and add them into HDF5 file 

    img : SimpleITK image to be pre-processed
    pm : prostate mask as SimpleITK image to be preprocesed
    resample_spacing : spacing to which the image have to be resampled 
    crop_size : x,y,z crop size in mm 
    intensitystandardize : as a tuple (template image as SimpleITK image, Number of histogram levels, Number of match points)
    '''
    
    # dilate the prostate mask and get an array of the pm
    pm = sitk.BinaryDilate(pm, 2, sitk.sitkBall)
    pmarr = sitk.GetArrayFromImage(pm)

    # initialize the names and labels list
    names = [] 
    labels = []

    # normalize the t2w and adc images (normalization not needed so just get image arrays)
    t2warr = sitk.GetArrayFromImage(t2w) # t2warr = normalizeImage(t2w,t2wmin,t2wmax,clipValue=t2wmax)
    adcarr = sitk.GetArrayFromImage(adc) # adcarr = normalizeImage(adc,adcmin,adcmax,clipValue=3000)

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
            peri = binary_dilation(slc, disk(dilate))
        else:
            peri = slc

        peri = peri * pmslc
        peri = peri.astype(np.uint8)
        props = regionprops(peri)

        if not peri.sum() == 0:

            # get the x and y coordinates of the mask
            sample = np.zeros((3,patchSize[1],patchSize[0]))
            starty, startx, endy, endx = props[0].bbox 
            t2wsample = t2warr[i, starty:endy, startx:endx]
            adcsample = adcarr[i, starty:endy, startx:endx]
            mask  = lsarr[i, starty:endy, startx:endx]
            orgmask = slc[starty:endy, startx:endx]

            # get t2w, adc image where the lesion overlays
            sample[0] = skresize(t2wsample,(patchSize[1],patchSize[0]),order=1)
            sample[1] = skresize(adcsample,(patchSize[1],patchSize[0]),order=1)
            sample[2] = sitk.GetArrayFromImage(DataUtil.resampleimagebysize(sitk.GetImageFromArray(mask),(patchSize[1],patchSize[0]),sitk.sitkNearestNeighbor))
            
            # append the name and lable to their respective list
            names.append(f'{name}_{i}')
            labels.append(label)
            _addToHDF5(sample, phase, outputpathname)
            
    # return the names and labels
    return names, labels

def _addToHDF5(sample, phase, outpath):
    '''
    imgarr : input image sample (ex 2 slices of the image)
    maskarr : output mask (ex. lesion segmentation mask)
    phase : phase of that image (train,test,val)
    '''
    
    # set the output folder and open the necessary hdf5 file
    outputfolder = outpath
    hdf5_file = tables.open_file(f'{outputfolder}/{phase}.h5', mode='a')

    # add the image to the hdf5 file and close
    data = hdf5_file.root['data']
    data.append(sample[None])
    hdf5_file.close()

# ----- normalization methods ----- #
# note that the below methods are not used here because...
# standardization has already been done for T2W images
# ADC images are quantitative and therefore should not be normalized
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

# ----- reporting spread of data ----- #
class ProstateDatasetHDF5(Dataset):
    '''
    @author Ansh Roge
    a class to define a dataset for easy loading a torch dataLoader
    '''

    def __init__(self, fname,transforms = None):
        self.fname=fname
        self.file = tables.open_file(fname)
        self.tables = self.file.root
        self.nitems=self.tables.data.shape[0]
        
        self.file.close()
        self.data = None
        self.mask = None
        self.names = None
        self.labels = None 
        self.transforms = transforms
         
    def __getitem__(self, index):
                
        self.file = tables.open_file(self.fname)
        self.tables = self.file.root
        self.data = self.tables.data
        self.labels = self.tables.labels
                
        if "names" in self.tables:
            self.names = self.tables.names

        data = self.data[index,:,:,:]   
        img = data[(0,1,2),:,:]
        
        if self.names is not None:
            name = self.names[index]
        
        label = self.labels[index]
        self.file.close()
        
        out = torch.from_numpy(img[None])
        
        return out,label,name

    def __len__(self):
        return self.nitems

def get_data(path, bs, phases):
    '''
    from the hdf5 file, split the data into a dataLoader with it's different phases
    - train, val, test
    '''
    
    # notify
    print('SEPARATING THE DATA FROM HDF5 FILES INTO DATALOADER OBJECTS')

    # define the returning variables
    dataLoader = {}
    dataLabels = {}

    nw = 2 # num workers
    # iterate through the phases
    for phase in phases:
        # read the h5 file according to the phase and isolate the labels and store
        filename = f'{path}/{phase}.h5'
        file = h5py.File(filename, libver='latest', mode='r')
        labels = np.array(file['labels'])
        file.close()
        dataLabels[phase] = labels

        # generate the loader and store
        data = ProstateDatasetHDF5(filename)
        loader = torch.utils.data.DataLoader(dataset=data, batch_size=bs, num_workers=nw, shuffle=True)
        dataLoader[phase] = loader

    # return the loader and labels
    return dataLoader, dataLabels    

def data_check(dataLoader, dataLabels, phases, labels):
    '''
    check the data for:
    - the train/val/test split
    - the class distribution
    '''
    
    # notify
    print('CHECKING THE SPREAD OF THE DATA')

    # print the size of the train, val, test datasets for the user
    phase_n = {}
    phase_label_count = {}
    total_n = 0
    for phase in phases:
        # count per phase
        phase_n[phase] = len(dataLoader[phase].dataset)

        # calculate total n
        total_n += len(dataLoader[phase].dataset)

        # count per label
        for label in labels:
            phase_label_count[f'{phase}_{label}'] = np.count_nonzero(dataLabels[phase] == label)

    # print for the user
    print(f'this is a {phase_n["train"] / total_n} / {phase_n["val"] / total_n} / {phase_n["test"] / total_n} for train / val / test splits')
    print(f'train set [n: {phase_n["train"]}, class 0%: {phase_label_count["train_0"] / phase_n["train"]}, class 1%: {phase_label_count["train_1"] / phase_n["train"]}]')
    print(f'val set: [n: {phase_n["val"]}, class 0%: {phase_label_count["val_0"] / phase_n["val"]}, class 1%: {phase_label_count["val_1"] / phase_n["val"]}]')
    print(f'test set: [n: {phase_n["test"]}, class 0%: {phase_label_count["test_0"] / phase_n["test"]}, class 1%: {phase_label_count["test_1"] / phase_n["test"]}]')

# ----- MAIN METHOD ----- #
def main():
    '''
    main method to generate HDF5 files from the data
    '''

    # read in the csv with the GGG scores (labels)
    labelsdf = pd.read_csv(labels_csv)

    # set dilate to none
    dilate = None 

    for rad in rads:
        print(rad)

        # set the variables used
        inputfoldername = f'{source_dir}/data/{race}_ggg-confirmed'
        newsize2D = (224,224) 
        depth = 0

        # read in the templates
        templateimg = sitk.ReadImage(t2w_template)
        templatepm = sitk.ReadImage(pm_template)
        templatepm = DataUtil.convert2binary(templatepm)
        
        # read in the splits and save as test, val, or train
        splitsdict = DataUtil.readJson(splitspath)
        cases = list(splitsdict.keys())

        # set the folder name to store the hdf5 files and generate
        hdf5path = f'rdp-hdf5/hdf5-{rad}'
        createHDF5(hdf5path, splitsdict, newsize2D, depth)
        
        # create a dictionary for the cases and labels
        casenames = {} 
        casenames['train'] = [] 
        casenames['val'] = []
        casenames['test'] = [] 
        caselabels = {} 
        caselabels['train'] = [] 
        caselabels['val'] = [] 
        caselabels['test'] = [] 

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
            dataset,pat = name.split('-')
            patname = f'{dataset}-{pat}'
            sb = Path(f'{inputfoldername}/{patname}')
            
            # get lesion mask paths - if T2W/ADC specific lesions exist, use those, otherwise use LS1
            lesionpath = Path(f'{source_dir}/data/{race}_lesion-masks/{patname}/')
            if len(glob.glob(f'{lesionpath}/T2W_LS.nii.gz')) > 0:
                lsfiles = glob.glob(f'{lesionpath}/T2W_LS.nii.gz')
            else:
                lsfiles = glob.glob(f'{lesionpath}/LS1.nii.gz')
            
            # Lesion loop
            outputfolder = f'{source_dir}/model-outputs-{race}/{hdf5path}'
            for k, lsfile in enumerate(lsfiles):
                lsfile = str(lsfile)
                lesion = lsfile.split('/')[-1].split('.')[0]
                print(f'Name: {patname}, Lesion: {lesion}, Progress: {j/len(cases)}')
                
                # set the severity according to GGG score (binary classification)
                label = 1 if (labelsdf[labelsdf['PatientID'] == f'prostate-{pat}']['GGG'] > 1.0).bool() else 0

                # apply augmentations to the minorty and majority classes to make class distribution more equal
                nosamples = num_augs[1] if label == 1 else num_augs[0]
                phase = splitsdict[name]
                if phase == 'train':
                    nosamples = 5 if label == 1 else 18
                    ret = getAugmentedData(sb, lsfile, nosamples=nosamples)
                elif phase == 'val':
                    ret = getAugmentedData(sb, lsfile, nosamples=nosamples)
                else:
                    nosamples = 15
                    ret = getAugmentedData(sb, lsfile, nosamples=nosamples)

                # add each aaugmented set of images to the HDF5 file
                for k,aug in enumerate(ret):
                
                    augt2w = aug[0][0]
                    augadc = aug[0][1]
                    augpm = aug[1][0]
                    augls = aug[1][1]
                    
                    casename = f'{name}_L{lesion}' if k == 0 else f'{patname}_L{lesion}_A{k}'

                    names,labels = addToHDF5(augt2w,augadc,augpm,augls,phase,hdf5path,newsize2D,t2wmin,t2wmax,adcmin,adcmax,casename,label,outputfolder,dilate=dilate)

                    casenames[phase].extend(names)
                    caselabels[phase].extend(labels)

        # add the file names and labels to the HDF5 file
        for phase in ['train', 'val', 'test']:
            hdf5_file = tables.open_file(f'{outputfolder}/{phase}.h5', mode='a')
            hdf5_file.create_array(hdf5_file.root, 'names', casenames[phase])
            hdf5_file.create_array(hdf5_file.root, 'labels', caselabels[phase])

            hdf5_file.close()

        # give the user the data split
        phases = ['train', 'val', 'test']
        batch_size = 1
        dataLoader, dataLabels = get_data(outputfolder, batch_size, phases)
        data_check(dataLoader, dataLabels, phases, labels = [0, 1])

if __name__ == '__main__':
    main()