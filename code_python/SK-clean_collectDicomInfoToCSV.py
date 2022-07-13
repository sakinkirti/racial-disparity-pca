import sys
import os 
from pathlib import Path
sys.path.append(str(Path(f"../Code_general")))
from sqlalchemy import true
from dataUtil import DataUtil 
from medImageProcessingUtil import MedImageProcessingUtil
import SimpleITK as sitk 
import shutil 
import pandas as pd 
import pydicom
from pydicom.tag import Tag 
import glob 

def main():
    # list of folder to get dicom info from
    dcm_path = ['/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/RDCasesFilteredAll/batch2_restructured/prostate_id 000722a/19000101/t2_tse_tra_320_p2',
                '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/RDCasesFilteredAll/batch2_restructured/prostate_id 000451/19000101/diff_tra-fov=240',
                '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/RDCasesFilteredAll/batch2_restructured/prostate_id 000241/19000101/t2_tse_tra_320_p2',
                '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/RDCasesFilteredAll/batch2_restructured/prostate_id 000166/19000101/t2_tse_tra_320_p2',
                '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/RDCasesFilteredAll/batch2_restructured/prostate_id 000104/19000101/t2_tse_tra_320_p2',
                '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/RDCasesFilteredAll/batch2_restructured/prostate_id 000030/19000101/t2_tse_tra_320_p2']
    dcm_path = [str(Path(folder)) for folder in dcm_path] 
    info = []
    
    # iterate through each path
    for path in dcm_path:
        # folder to store the data in
        subdirs = DataUtil.getSubDirectories(path)

        # iterate through each directory
        for dir in subdirs:
            # set the image directories
            input = str(dir)
            dcms = glob.glob(input + '**/*.dcm', recursive=True)

            # iterate through the folders
            for dcm in dcms:
                # get the first .dcm file in each folder
                img = pydicom.read_file(dcm)

                # get the series id, sop id, and
                id_tag = Tag(0x00100020)
                sop_tag = Tag(0x00080018)
                ser_tag = Tag(0x0020000e)
                id = img[id_tag].value
                sop = img[sop_tag].value
                ser = img[ser_tag].value

                info.append([id, str(dir).split('/')[-1], ser, sop])

            # print to the console
            print("data saved for image stack: " + str(dir).split('/')[-1])

    # output to csv
    df = pd.DataFrame(info,columns=['PatientID','Image','Series','SOP'])
    df.to_csv(f'dicom_info.csv',index=None)
    print("csv exported and saved.")

if __name__ == '__main__':
    main()