import sys
import os
from pathlib import Path

sys.path.append(str(Path("C:\\Users\\Jayas\\Downloads")))
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
    inputfolders = [f'/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/RDCasesFilteredAll/batch2_restructured2']

    inputfolders = [str(Path(x)) for x in inputfolders]

    info = []

    for inputfolder in inputfolders:
        subdirs = DataUtil.getSubDirectories(inputfolder)

        for sb in subdirs:
            print(sb.stem)

            # sb= "C:\\Abhishek\\Research\\CaseWestern\\DataHandling\\RestructuredFinal\\TPa\\prostate_002018a\\19000101\\"
            inpath = str(sb)

            # check the number of subdirectories are present and change this accordingly
            files = glob.glob(inpath + '/**/**/**/*.dcm', recursive=True)
            # files = glob.glob(inpath + '/**/**/*.dcm', recursive=True)

            for f in files:
                # print(files.index(f))
                # if files.index(f)==48:
                #     print("I am here")
                ds = pydicom.read_file(f)
                sopT = Tag(0x00080018)
                serT = Tag(0x0020000e)
                stuT = Tag(0x0020000d)
                insT = Tag(0x00200013)
                zspacingT = Tag(0x00180088)
                inplanespacingT = Tag(0x00280030)
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
                    xspacing, yspacing = ds[inplanespacingT].value

                    xspacing = float(xspacing)
                    yspacing = float(yspacing)

                    imageorientation = ds[imageorgientationT].value
                    imageorientation.extend([0, 0, 1])

                    imageorigin = ds[imageoriginT].value

                    patientid = ds[patientidT].value

                    protocolname = ds[protocolnameT].value

                    info.append([patientid, stu, ser, protocolname, sop, ins, xspacing, yspacing, zspacing, imageorigin,
                                 imageorientation])

                except:
                    print(f)

    df = pd.DataFrame(info,
                      columns=['PatientID', 'Study', "Series", 'Protocol', 'SOP', 'Instance', 'Xspacing', 'Yspacing',
                               'Zspacing', 'Origin', 'Orientation'])

    df['TotalSlices'] = df.groupby('Series')['Instance'].transform('max')
    df.to_csv('/Users/sakinkirti/Programming/Python/CCIPD/Code/dicom_info_batch2.csv', index=None)







