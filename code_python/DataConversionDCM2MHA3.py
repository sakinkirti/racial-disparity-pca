import os
import sys
import numpy as np
import pydicom as pd
import natsort
import SimpleITK as sitk
import pandas as pd1
from pydicom.tag import Tag


def dcm2mha(DicomPath, outFileLoc, outFileName):
    DicomList = [f for f in os.listdir(DicomPath) if (os.path.isfile(os.path.join(DicomPath, f)))]
    DicomList = [x for x in DicomList if x.endswith('.dcm')]
    DicomList = natsort.natsorted(DicomList, reverse=False)
    if len(DicomList)>1:
        # To get first image as refenece image, supposed all images have same dimensions
        ReferenceImage = pd.dcmread(DicomPath +"\\"+ DicomList[len(DicomList)-1])
        ReferenceImage1 = pd.dcmread(DicomPath +"\\"+  DicomList[len(DicomList)-2])
        # To get Dimensions
        Dimension = (int(ReferenceImage.Rows), int(ReferenceImage.Columns), len(DicomList))
        # To get 3D spacing
        Spacing = (
        float(ReferenceImage.PixelSpacing[0]), float(ReferenceImage.PixelSpacing[1]), abs(float(ReferenceImage1.ImagePositionPatient[2]-ReferenceImage.ImagePositionPatient[2])))
        # To get image origin
        Origin = ReferenceImage.ImagePositionPatient
        imageorgientationT = Tag(0x00200037)
        Orientation = ReferenceImage[imageorgientationT].value
        Orientation.extend([0, 0, 1])
        # To make numpy array
        NpArrDc = np.zeros(Dimension, dtype=ReferenceImage.pixel_array.dtype)
        # loop through all the DICOM files
        if len(ReferenceImage.pixel_array.shape)<3:
            for filename in DicomList:
                df = pd.dcmread(DicomPath+"\\" + filename)
                # store the raw image data
                NpArrDc[:, :, DicomList.index(filename)] = (df.pixel_array)
            NpArrDc = np.transpose(NpArrDc, (2, 0, 1))
            sitk_img = sitk.GetImageFromArray(NpArrDc, isVector=False)
            sitk_img.SetSpacing(Spacing)
            sitk_img.SetOrigin(Origin)
            sitk_img.SetDirection(Orientation)
            if not os.path.exists(outFileLoc):
                os.makedirs(outFileLoc)
            # sitk.WriteImage(sitk_img, (outFileLoc+"\\"+outFileName+".mha"))
            sitk.WriteImage(sitk_img, (outFileLoc + "\\"+outFileName + ".nii.gz"))



if __name__ == "__main__":
    Folder_Path = "G:\\My Drive\\Prostate Data\\UH_Prostate_Radiology\\UH_Serial_MRI\\Restructured\\Resturctured_Midya_Final\\TP1_Final"
    NIFTIPath = "G:\\My Drive\\Prostate Data\\UH_Prostate_Radiology\\UH_Serial_MRI\\NIFTI2\\TP1-Set2\\"
    PatientList = [p for p in sorted(os.listdir(Folder_Path)) if os.path.isdir(os.path.join(Folder_Path, p))]
    dataCSV = pd1.read_csv('C:\\Users\\Jayas\\Downloads\\mdai_dasa_project_MmRaMRrb_annotations_dataset_D_Yy3vXx_2022-03-01-175000.csv')
    SeriesInstanceList=dataCSV.SeriesInstanceUID
    for i in range(18,len(PatientList)):
        pt= PatientList[i]
        print(i,"--->", pt)
        if pt.find("_")> -1:
            ptId= pt[pt.find("_")+4:]
        else:
            ptId=pt
        #pt in PatientList:
        PtFolderList= [f for f in sorted(os.listdir(Folder_Path+"\\"+pt))]
        if pt.upper().find("PROSTATE")>-1:

            if pt.find("-")>-1:
                outFileName=pt[0:pt.find("-")]
            else:
                outFileName=pt
            pt=pt.zfill(6)
            outFileName=outFileName.zfill(6)
            outFileName.replace('prostate_id ', 'prostate_id_')

        else:
            outFileName="prostate_id_"+pt

        for ptF in PtFolderList:
            if ptF.upper().find("ADC")>-1:
                SubFolderList = [f for f in os.listdir(Folder_Path + "\\" + pt + "\\" + ptF + "\\") if os.path.isdir(Folder_Path + "\\" + pt + "\\" + ptF + "\\" + f) == True]
                for sf in SubFolderList:
                    DicomPath=  Folder_Path+"\\"+pt+"\\"+ptF+"\\"+sf
                    dcm2mha(DicomPath, NIFTIPath+"\\" + outFileName , outFileName + "_ADC"+"_"+ptF)
            elif ptF.upper().find("T2")>-1:
                SubFolderList=[ f for f in os.listdir(Folder_Path+"\\"+pt+"\\"+ptF+"\\") if os.path.isdir(Folder_Path+"\\"+pt+"\\"+ptF+"\\"+f)== True]
                for sf in SubFolderList:
                    DicomPath=Folder_Path+"\\"+pt+"\\"+ptF+"\\"+sf
                    dicomList= [ f for f in os.listdir(DicomPath) if f.endswith(".dcm")]
                    dicomList = natsort.natsorted(dicomList, reverse=False)
                    df= pd.dcmread(Folder_Path+"\\"+pt+"\\"+ptF+"\\"+sf+"\\"+dicomList[0])
                    SerId= df.SeriesInstanceUID
                    if SerId in SeriesInstanceList.values:
                        dcm2mha(DicomPath, NIFTIPath+"\\" +outFileName , outFileName + "_T2")













