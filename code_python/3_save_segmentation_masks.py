from tokenize import group
import pandas as pd
import numpy as np
import yaml
import cv2
import SimpleITK as sitk
import ast
import sys
import os
from pathlib import Path
sys.path.append(str(Path(f"C:\\Users\\Jayas\\Downloads")))
from dataUtil import DataUtil

# enter the json file name without extenion and dicom filename without extension
filename = '/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/dicom and json/check_json_sree'
dicomfilename = '/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/dicom and json/dicom_info'

# Set the output path below (Segmentations will be saved here)
outputfolder = '/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/to_nifti_cleaned'
DataUtil.mkdir(outputfolder)

df = pd.read_csv(f'{filename}.csv')

df = df[df['annotationMode']=='freeform']

dicomdf = pd.read_csv(f'{dicomfilename}.csv')

groupby = df.groupby(['SeriesInstanceUID','LabelName'])

labeldict = {}
labeldict['PI-RADS 3'] = 3
labeldict['PI-RADS 4'] = 4
labeldict['PI-RADS 5'] = 5

series_prev = df['SeriesInstanceUID'].values[0]
counter = 0
for key,groupdf in groupby:

    series, label = key
    if series == series_prev:
        counter += 1
    else:
        counter = 1
    series_prev = series

    totalslices = dicomdf[dicomdf['Series']==series]['TotalSlices'].values.max()
    xspacing = dicomdf[dicomdf['Series']==series]['Xspacing'].values[0]
    yspacing = dicomdf[dicomdf['Series']==series]['Yspacing'].values[0]
    zspacing = dicomdf[dicomdf['Series']==series]['Zspacing'].values[0]


    origins = dicomdf[dicomdf['Series']==series]['Origin'].values
    origins = [tuple(ast.literal_eval(x)) for x in origins]
    origins = list(set(origins))
    origins.sort(key=lambda x:x[2])



    # import pdb
    # pdb.set_trace()

    origin = origins[0]
    direction = ast.literal_eval(dicomdf[dicomdf['Series']==series]['Orientation'].values[0]) #please check!!
    patientid = dicomdf[dicomdf['Series']==series]['PatientID'].values[0]




    sizex = None
    sizey = None
    maskarr = None

    for i,row in enumerate(groupdf.iterrows()):

        if i == 0:
            sizex = row[1]['height']
            sizey = row[1]['width']

            maskarr = np.zeros((int(totalslices),int(sizey),int(sizex)))

        sop = row[1]['SOPInstanceUID']

        instance = dicomdf[dicomdf['SOP']==sop]['Instance'].values[0]
        vertices = yaml.load(row[1]['data'])['vertices']

        cv2.fillPoly(maskarr[instance], np.int32([vertices]), (255, 255, 255))


    if maskarr is not None:

        maskarr[maskarr > 0] = labeldict[label]

        mask = sitk.GetImageFromArray(maskarr)

        mask.SetOrigin(tuple(origin))
        Spacing=(xspacing,yspacing,float(zspacing))
        mask.SetSpacing(Spacing)
        mask.SetDirection(tuple(direction))
        # print('done')

        path = f'{outputfolder}/prostate-{str(patientid).split("_")[-1]}'
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'created dir: {path}')
        sitk.WriteImage(mask,str(Path(f'{path}/LS{counter}_ST.nii.gz')))
        print(patientid)


