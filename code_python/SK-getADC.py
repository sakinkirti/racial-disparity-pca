import sys
from pathlib import Path
sys.path.append(str(Path(f"../Code_general")))
from dataUtil import DataUtil 
import SimpleITK as sitk 
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import glob 

'''
@author Sakin Kirti
@date 5/15/2022

script to gather the ADC images from a distinct set of ADC images
'''

# the large directory storing folders named with each prostate id to save
to_gather = '/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/to_nifti_cleaned'

def main():
    # get a list of prostate ids and iterate through them
    prostate_ids = DataUtil.getSubDirectories(to_gather)
    for id in prostate_ids:
        # rename each id to be the right name in the search dir
        search_id = 'prostate_id ' + str(id).split('-')[-1]

        # find the adc image in the original batch
        ADC_path = glob.glob(f'/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/all_batch/{search_id}/**/*adc*/**', recursive=True)
        if ADC_path == []:
            ADC_path = glob.glob(f'/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/all_batch/{search_id}/**/*ADC*/**', recursive=True)

        # convert the ADC image in the path to a nifti image and save it in the to_gather folder under its prostate id
        out_dir = id
        DataUtil.dicom_to_nifti(ADC_path[1], f'{out_dir}/ADC.nii.gz')
        print(f'completed image {str(id).split("/")[-1]}')

if __name__ == '__main__':
    main()    