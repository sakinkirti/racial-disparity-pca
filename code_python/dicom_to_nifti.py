import os
import sys
sys.path.append('sourcecode/classification_lesion/Code_general')
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from dataUtil import DataUtil

def main():
    dcm_path = "/Users/anshroge/Downloads/"
    pats = DataUtil.getSubDirectories(dcm_path)
    pat = pats[0]
    patid = str(pat).split('/')[-1]
    
    output = fr"/Users/anshroge/Downloads/{patid}.nii.gz"
    
    DataUtil.dicom_to_nifti(pat, output)

if __name__ == "__main__":
    main()