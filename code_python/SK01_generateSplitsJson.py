import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import cv2
from skimage.measure import find_contours
from sklearn import metrics
import sys
import itertools
from tqdm import tqdm
from statUtil import StatUtil
from dataUtil import DataUtil
from glob import glob
import seaborn as sns
import json
from skimage import measure
from sklearn.model_selection import train_test_split
import random

'''
@author Sakin Kirti
@date 6/11/22

script to generate a json file that labels each training sample as train, val, test
'''

# path to spreadsheets containing GGG scores
aa_csv_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-AA/racial_disparity_FINAL_batch1_anonymized.xlsx'
ca_csv_path = ['/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-CA/ClinicalInfoFinal_AS.xlsx', 
                '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-CA/racial_disparity_FINAL_batch1_anonymized.xlsx',
                '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-CA/UH_BCR_clinical dat.xlsx']

# where to save the json splits
aa_json_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-AA/rdp-splits-json/racial-disparity-splits.json'
ca_json_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-CA/rdp-splits-json/racial-disparity-splits.json'

# the specifc race to generate splits for (AA, CA, or RA)
race = 'AA'

# random seed and the proportion to split the data into
rs = 1234
split_prop = {'train' : 0.49, 
                'val' : 0.21 / 0.51, 
                'test' : 0.3 / 0.3}

def generate_splits_AA():
    '''
    method to generate splits for the specifically african american patients
    uses the aa_csv_path'''

    # read the csv file
    df = pd.read_excel(aa_csv_path)
    df = df[df['GGG (1-5)'].notnull()]
    
    sigs = {}
    for index, row in df.iterrows():
        patnum = row['ID'].split(' ')[-1]

        # split the data according to GGG
        sigs[f'prostate-{patnum}'] = 1 if row['GGG (1-5)'] > 1 else 0

    # conver to df
    sigs_df = pd.DataFrame(sigs.items(), columns=['ID', 'Sig'])

    # train test split
    train, val = train_test_split(sigs_df, test_size=0.51, random_state=rs)
    val, test = train_test_split(val, test_size=0.59, random_state=rs)

    # let the user know what the splits look like
    print(f'this is a {train.shape[0] / df.shape[0]} / {val.shape[0] / df.shape[0]} / {test.shape[0] / df.shape[0]} for train / val / test splits')
    print(f'train set [n: {train.shape[0]}, class 0%: {train["Sig"].value_counts()[0] / train.shape[0]}, class 1%: {train["Sig"].value_counts()[1] / train.shape[0]}]')
    print(f'val set [n: {val.shape[0]}, class 0%: {val["Sig"].value_counts()[0] / val.shape[0]}, class 1%: {val["Sig"].value_counts()[1] / val.shape[0]}]')
    print(f'test set [n: {test.shape[0]}, class 0%: {test["Sig"].value_counts()[0] / test.shape[0]}, class 1%: {test["Sig"].value_counts()[1] / test.shape[0]}]')
    
    split = {}
    for index, row in train.iterrows():
        split[row['ID']] = "train"
    for index, row in val.iterrows():
        split[row['ID']] = "val"
    for index, row in test.iterrows():
        split[row['ID']] = "test"
        
    with open(aa_json_path, 'w') as outfile:
        json.dump(split, outfile)

def generate_splits_CA():
    '''
    generate splits for the caucaisan american patients
    '''

    # store the patients with class 0 and 1
    sigs_0 = {}
    sigs_1 = {}

    # ------------------------ read the first csv file
    df = pd.read_excel(ca_csv_path[0])
    df = df[df['Initi GS1'].notnull()]
    
    for index, row in df.iterrows():
        patnum = row['ImgName'].split('_')[-1]

        # split the data according to GGG
        if row['Initi GS1'] > 1:
            sigs_1[f'prostate-{patnum}'] = 1
        else:
            sigs_0[f'prostate-{patnum}'] = 0

    # ------------------------ read the second csv file
    df = pd.read_excel(ca_csv_path[1])
    df = df[df['GGG (1-5)'].notnull()]
    
    # append Gleeson scores onto the sigs1, sigs0 dictionaries
    for index, row in df.iterrows():
        patnum = row['ID'].split(' ')[-1]

        # split the data according to GGG
        if row['GGG (1-5)'] > 1:
            sigs_1[f'prostate-{patnum}'] = 1
        else:
            sigs_0[f'prostate-{patnum}'] = 0

    # ------------------------ read the third csv file
    df = pd.read_excel(ca_csv_path[2])
    df = df[df['bGG'].notnull()]
    
    # append Gleeson scores onto the sigs1, sigs0 dictionaries
    for index, row in df.iterrows():
        patnum = row['pID'].zfill(6)

        # split the data according to GGG
        if row['bGG'] > 1:
            sigs_1[f'prostate-{patnum}'] = 1
        else:
            sigs_0[f'prostate-{patnum}'] = 0
    
    # convert each dictionary to a pandas df
    sigs_0_df = pd.DataFrame(sigs_0.items(), columns=['ID', 'Sig'])
    sigs_1_df = pd.DataFrame(sigs_1.items(), columns=['ID', 'Sig'])

    # train, val, test split manually
    train = pd.concat( [sigs_0_df.sample(frac=split_prop['train'], replace=False, random_state=rs), sigs_1_df.sample(frac=split_prop['train'], replace=False, random_state=rs)], axis=0 )
    val = pd.concat( [sigs_0_df.sample(frac=split_prop['val'], replace=False, random_state=rs), sigs_1_df.sample(frac=split_prop['val'], replace=False, random_state=rs)], axis=0 )
    test = pd.concat( [sigs_0_df.sample(frac=split_prop['test'], replace=False, random_state=rs), sigs_1_df.sample(frac=split_prop['test'], replace=False, random_state=rs)], axis=0 )

    # let the user know what the splits look like
    print(f'this is a {train.shape[0] / df.shape[0]} / {val.shape[0] / df.shape[0]} / {test.shape[0] / df.shape[0]} for train / val / test splits')
    print(f'train set [n: {train.shape[0]}, class 0%: {train["Sig"].value_counts()[0] / train.shape[0]}, class 1%: {train["Sig"].value_counts()[1] / train.shape[0]}]')
    print(f'val set [n: {val.shape[0]}, class 0%: {val["Sig"].value_counts()[0] / val.shape[0]}, class 1%: {val["Sig"].value_counts()[1] / val.shape[0]}]')
    print(f'test set [n: {test.shape[0]}, class 0%: {test["Sig"].value_counts()[0] / test.shape[0]}, class 1%: {test["Sig"].value_counts()[1] / test.shape[0]}]')
    
    split = {}
    for index, row in train.iterrows():
        split[row['ID']] = "train"
    for index, row in val.iterrows():
        split[row['ID']] = "val"
    for index, row in test.iterrows():
        split[row['ID']] = "test"
        
    with open(ca_json_path, 'w') as outfile:
        json.dump(split, outfile)

def generate_splits_RA():
    '''
    method to generate splits for a race agnostic model
    ie taking both african american and caucasian american patients
    similar method as the CA where each spreadsheet is added individually
    '''

def main(race):
    '''
    differentiates between which split should be created and call that specific method
    '''

    if race == 'AA':
        generate_splits_AA()
    elif race == 'CA':
        generate_splits_CA()
    elif race == 'RA':
        generate_splits_RA()
    
if __name__ == '__main__':
    main(race)