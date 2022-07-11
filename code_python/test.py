import os
import glob
import SimpleITK as sitk

from dataUtil import DataUtil as DU
from imageDataUtil import ImageData as IDU

path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/RA_RDCasesFilteredAll'

dirs = DU.getSubDirectories(dir=path)

for patient in dirs:
    image = glob.glob(pathname=f'{patient}/T2W_orig.nii.gz')

    for img in image:
        image = sitk.ReadImage(img)
        origin, spacing, direction = IDU.get_image_data(image)
        
        if origin != (0, 0, 0):
            print(f'{str(patient).split("/")[-1]}')