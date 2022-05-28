# racial-disparity-pca

@Author Sakin Kirti CWRU Dept of CDS 2023
repository storing code and data for deep learning on 3D prostate cancer MRI

## code_matlab
Contains...
1. elastix program for registration of ADC images onto T2W images (elastix folder)
2. matlab script to loop through images and perform registration (elastixRegistration.m)
3. file that tells elastix how to perform the registration (rigid.txt)
4. short script to view a specified ADC image output (viewSlice.m)

## code_python
Contains all scripts for the associated project. The order they are used in is...
1. 1_mdai_json_to_csv.py to create a csv containing the vertices of the lesion masks
2. 2_collect_dicom_info_to_csv.py to create a csv containing information about the dicom slices
3. 3_save_segmentation_masks.py to create and save the lesion masks as nifti files
4. SK-dicomToNifti.py to create nifti files of each dicom series
5. SK-getADC.py to get and store the ADC image files
6. SK-saveImagesWithOverlay to create overlays showing the lesion mask on the T2W image
7. 

## dataset
Contains the data used in this project
1. clean - the original nifti T2W, ADC, and mask files
2. dicom_json - the dicom_info csv and json info to create the lesion masks
3. mask_overlays - overlays of the lesions on top of each T2W file

## procedure

