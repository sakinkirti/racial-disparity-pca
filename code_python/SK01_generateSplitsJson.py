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

'''
@author Sakin Kirti
@date 6/11/22

script to generate a json file that labels each training sample as train, val, test
'''

# global vars
csv_path = "/Volumes/GoogleDrive/My Drive/racial-disparity-pc/racial_disparity_FINAL_batch1_anonymized.csv"
json_path = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/model-outputs/racial-disparity-splits.json'

rs = 1234
split_prop = {'train' : 0.49, 
                'val' : 0.21, 
                'test' : 0.3}

def main():
    # read the csv file
    df = pd.read_csv(csv_path)
    df = df[df['GGG (1-5)'].notnull()]
    
    sigs_0 = {}
    sigs_1 = {}
    for index, row in df.iterrows():
        patnum = row['ID'].split(' ')[-1]

        # split the data according to GGG
        if row['GGG (1-5)'] > 1:
            sigs_1[f'prostate-{patnum}'] = 1
        else:
            sigs_0[f'prostate-{patnum}'] = 0
    
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
        
    with open(json_path, 'w') as outfile:
        json.dump(split, outfile)
    
if __name__ == '__main__':
    main()