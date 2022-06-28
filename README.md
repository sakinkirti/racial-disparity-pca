# racial-disparity-pca

@author Sakin Kirti CWRU Dept of CDS 2023 <br>
repository storing code and data for deep learning on 3D prostate cancer MRI

## code_matlab
Contains...
1. elastix program for registration of ADC images onto T2W images (elastix folder)
2. matlab script to loop through images and perform registration (elastixRegistration.m)
3. file that tells elastix how to perform the registration (rigid.txt)
4. short script to view a specified image output (viewSlice.m)

## code_python
Contains all scripts for the associated project. Note that there are many other scripts in this folder. They are used to clean the data (make sure it is in the right orientation, with the right number of slices, etc and were used between steps 4 and 5). The order they are used in is...
1. 1_mdai_json_to_csv.py to create a csv containing the vertices of the lesion masks
2. 2_collect_dicom_info_to_csv.py to create a csv containing information about the dicom slices
3. 3_save_segmentation_masks.py to create and save the lesion masks as nifti files
4. SK-dicomToNifti.py to create nifti files of each dicom series
5. SK-getADC.py to get and store the ADC image files
6. SK-saveImagesWithOverlay to create overlays showing the lesion mask on the T2W image
7. registration and standardization -> performed using matlab scripts (refer to code_matlab)
8. SK-prostateSegmentation.py to segment the prostate gland and transition zone from the MRI -> need to have proSeg folder with proSeg.exe to run
9. SK-filterGGG.py filters the data collected to just the ones that we have documented GGG (gleeson grade) scores for
10. SK_04generatePredictions.py was used along with one of @AnshRoge's prebuilt models to see if directly using that network would work well
11. SK_00racialDisparitySqueezeNet.py details the SqueezeNet architecture used for training a new model
12. SK_01generateSplitsJson.py creates a json file by randomly dividing the spreadsheet with GGG scores into train, val, and test sets
13. SK_02pathAugmentSaveHDF5.py creates patches right around the cancer lesion, creates some augmentations of the data and saves them to HDF5 files of either train, val, or test
14. SK_03transferLearn.py uses the SqueezeNet architecture to train a new model on the GGG scores based on the input augmented patches
15. SK_04generatePredictions.py used to test the newly learned weights on the test set
15. SK_05generateActivationMaps.py to generate activation maps showing which regions of the patches were used in developing each prediction

## procedure

