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
Contains all scripts for the associated project. Note that there are many other scripts in this folder. many different scripts are used for this project. Each of their uses is prefixed by a semi-useful title...
- `clean` is used to prep the data, from individual dicom slices to nifti files as well as cleaning those nifti files
- `preprocess` is used to create any overlays and preprocessing that may be needed. Note that registration and standardization are also preprocessing steps, but are written in matlab
- `metadata` area used to organize the metadata of the nifti files
- `organize` are scripts meant to organize files and the file structure
- `SK00` - `SK05` are scripts used for creating HDF5 files and training and testing the SqueezeNet architecture

## procedure

