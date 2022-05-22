import sys
import os 
from pathlib import Path
sys.path.append(str(Path(f"../Code_general")))
from dataUtil import DataUtil 
from medImageProcessingUtil import MedImageProcessingUtil
import SimpleITK as sitk 
import shutil 
import pandas as pd 
import pydicom
from pydicom.tag import Tag 
import glob 


if __name__ == "__main__":

    # Enter the list of dicom folders you would like to collect the dicom information from 
    inputfolders = [f'/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/batch2_restructured']

    inputfolders = [ str(Path(x)) for x in inputfolders ]

    info = []

    for inputfolder in inputfolders:
        subdirs = DataUtil.getSubDirectories(inputfolder)

        for sb in subdirs:
            
            inpath = str(sb)

            # check the number of subdirectories are present and change this accordingly
            files = glob.glob(inpath + '/**/**/**/*.dcm', recursive=True)
            # files = glob.glob(inpath + '/**/**/*.dcm', recursive=True)

            for f in files:
                
                ds = pydicom.read_file(f) # read file in from this line, use the series instace UID - would be same for all files in that folder, and sop instance uid - different for each specific file in the folder
                sopT = Tag(0x00080018)
                serT = Tag(0x0020000e)
                stuT = Tag(0x0020000d)
                insT = Tag(0x00200013)
                zspacingT = Tag(0x00180088)
                inplanespacingT = Tag(0x00280030) # define the tag
                imageorgientationT = Tag(0x00200037)
                imageoriginT = Tag(0x00200032) 
                patientidT = Tag(0x00100020)
                protocolnameT = Tag(0x00181030)

                try:
                    sop = ds[sopT].value
                    ser = ds[serT].value
                    stu = ds[stuT].value
                    ins = int(ds[insT].value)
                    zspacing = float(ds[zspacingT].value)
                    xspacing,yspacing = ds[inplanespacingT].value # use the tag with the image

                    xspacing = float(xspacing)
                    yspacing = float(yspacing)

                    imageorientation = ds[imageorgientationT].value
                    imageorientation.extend([0,0,1])

                    imageorigin = ds[imageoriginT].value

                    patientid = ds[patientidT].value 

                    protocolname = ds[protocolnameT].value

                    info.append([patientid,stu,ser,protocolname,sop,ins,xspacing,yspacing,zspacing,imageorigin,imageorientation])

                except:
                    print(f) 


    df = pd.DataFrame(info,columns=['PatientID','Study',"Series",'Protocol','SOP','Instance','Xspacing','Yspacing','Zspacing','Origin','Orientation'])

    df['TotalSlices'] = df.groupby('Series')['Instance'].transform('max')

    df.to_csv('dicom_info.csv',index=None)







