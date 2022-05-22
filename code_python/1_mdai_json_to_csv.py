import sys
import os  
sys.path.append(fr"../Code_general")
from dataUtil import DataUtil 
from medImageProcessingUtil import MedImageProcessingUtil
import SimpleITK as sitk 
import shutil 
import pandas as pd 

def getLabelGroups(dct):

    df = None 

    for labelgroup in dct['labelGroups']:
        name = labelgroup['name']
        _df = pd.DataFrame.from_dict(labelgroup["labels"])
        _df['LabelGroup'] = name 
        df = _df if df is None else df.append(_df)

    return df 

def getStudies(dct):

    df = None 

    for dataset in dct['datasets']:
        name = dataset['name']
        description = dataset['description']
        sty = dataset['studies']
        _df = pd.DataFrame(sty)
        _df['StudyName'] = name 
        _df['StudyDescription'] = description

        df = _df if df is None else df.append(_df)

    return df 

def getAnnotations(dct):
    
    df = None 

    for dataset in dct['datasets']:
        name = dataset['name']
        description = dataset['description']
        ans = dataset['annotations']
        _df = pd.DataFrame(ans)
        _df['StudyName'] = name 
        _df['StudyDescription'] = description

        df = _df if df is None else df.append(_df)

    return df 


if __name__ == "__main__":

    # enter the file name of json without extenstion 
    filename = "/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/dicom and json/mdai_dasa_project_MmRaMRrb_annotations_dataset_D_DkXZEx_2022-04-24-172619"

    md = DataUtil.readJson(f'{filename}.json')

    lgsdf = getLabelGroups(md)
    lgsdf = lgsdf.rename(columns={"name":"LabelName","id":"labelId"})

    stydf = getStudies(md)
    ansdf = getAnnotations(md)

    m1 = ansdf.merge(stydf,on=['StudyInstanceUID','StudyName','StudyDescription'])
    m1 = m1.rename(columns={"number":"ExamNumber"})


    m1 = m1[['ExamNumber','labelId','StudyInstanceUID','SeriesInstanceUID','SOPInstanceUID','createdAt','updatedAt','height','width','data','note','StudyName']]
    lgsdf = lgsdf[['labelId','LabelName','LabelGroup','annotationMode']]

    m2 = m1.merge(lgsdf,on='labelId')

    m2 = m2[['ExamNumber','StudyInstanceUID', 'SeriesInstanceUID','SOPInstanceUID','labelId','LabelName','LabelGroup', 'annotationMode','createdAt',
        'updatedAt', 'height', 'width', 'data', 'note', 'StudyName',]]

    m2.to_csv(f'check_json_amogh.csv',index=None)




