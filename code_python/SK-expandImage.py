from unittest import result
import skimage.transform as skTrans
import nibabel as nib
import os
import SimpleITK as sitk

file_path = '/Users/sakinkirti/Downloads/drive-download-20220601T155025Z-001/AH_ADC_template.nii'
im = nib.load(file_path).get_fdata()
result1 = skTrans.resize(im, (320,320,28), order=1, preserve_range=True)


nib.save(result1, 'AH_ADC_template_resamp.nii')