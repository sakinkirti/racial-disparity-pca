from socketserver import DatagramRequestHandler
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np 
import pandas as pd
from skimage import measure
from skimage.measure import regionprops
import glob
import pydicom
from pydicom.tag import Tag
from sqlalchemy import true

from dataUtil import DataUtil

# get the image directories from the input path
main_path = '/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/all_batch'
check_json = '/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/dicom and json/check_json_sree.csv'
dicom_info = '/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/dicom and json/dicom_info.csv'
output_path = '/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/to_nifti_cleaned'

# image path
single_dcm = '/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/all_batch/prostate_id 000951/T2-AX/6'

def main(series):
    # if you want to convert just a single series (dcm stack) set the value to true in the main method
    if series:
        # convert the dicoms to nifti
        nii_path = f'{output_path}/prostate-000951/T2W.nii.gz'
        dicom_to_nifti(single_dcm, nii_path)
        print(f'processed img')
    else:
        # get the check_json - only convert those images which have annotations
        json_data = pd.read_csv(check_json)
        json_data = json_data[json_data['annotationMode'] == 'freeform']

        # get the dicom_info
        dicom_data = pd.read_csv(dicom_info)

        # iterate through the freeform series id
        group = json_data.groupby(['SeriesInstanceUID', 'LabelName'])
        for key, data in group:
            series_id, label = key

            # find the prostate_id in dicom_info
            prostate_id = 'prostate_id ' + str(dicom_data[dicom_data['Series'] == series_id]['PatientID'].values[0]).split('_')[-1]
            search_dir = f'{main_path}/{prostate_id}'
            dcm_folder = find_dcm_folder(search_dir, series_id)

            # convert the dicoms to nifti
            nii_path = f'{output_path}/prostate-{search_dir.split("/")[-1].split(" ")[-1]}/T2W.nii.gz'
            dicom_to_nifti(dcm_folder, nii_path)
            print(f'processed img {search_dir.split("/")[-1]}')

'''
create another dicom_to_nifti (the original is in dataUtil) 
function because need to reverse the file order
@param dir - the director in which the dicoms are held
@param output - the output path
'''
def dicom_to_nifti(dir, output):
    reader = sitk.ImageSeriesReader()
    dcm_names = list(reader.GetGDCMSeriesFileNames(str(dir)))
    dcm_names.reverse() # GetGDCMSeriesFileNames gives the list of images in the reverse order, so need to put them in the right order
    reader.SetFileNames(dcm_names)
        
    image = reader.Execute()
        
    sitk.WriteImage(image, output)

'''
function to find the folder that contains the dicoms of interest
@param main_directory - the directory containing all dicoms
@param series_id - the series_id that you are searching for
@return the filepath containing all dicoms of interest
'''
def find_dcm_folder(main_directory, series_id):
    # get the list of dicoms to search through
    main_directory = f'{main_directory}/'
    search_files = glob.glob(f'{main_directory}/**/**/**/*.dcm', recursive=True)
    series_tag = Tag(0x0020000e)

    # loop through the search_files and get the dicom series id
    for dcm in search_files:
        img = pydicom.read_file(dcm)

        if series_id == img[series_tag].value:
            dcm_path = ''
            for path in str(dcm).split('/')[0:-1]:
                dcm_path = dcm_path + '/' + path
            return dcm_path

if __name__ == '__main__':
    main(True)