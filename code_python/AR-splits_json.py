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



def main():
    px1csv = "dataset/racial_disparity_FINAL_batch1_anonymized.csv"
    df = pd.read_csv(px1csv)
    df = df[df['GGG (1-5)'].notnull()]
    # df = df.dropna()
    
    sigs = {}
    for index, row in df.iterrows():
        patnum = row['ID'].split(' ')[-1]
        # lsnum = str(row['fid']).split('.')[0]
        sigs[fr'prostate-{patnum}'] = 1 if row['GGG (1-5)'] > 1 else 0
        
    sigs_df = pd.DataFrame(sigs.items(), columns=["ID", "Sig"])
    
    train, test = train_test_split(sigs_df, test_size=0.33)
    
    split = {}
    
    for index, row in train.iterrows():
        split[row['ID']] = "train"
    for index, row in test.iterrows():
        split[row['ID']] = "test"
        
    
    jsonpath = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/model-outputs/racialdisparitysplits.json'
    with open(jsonpath, 'w') as outfile:
        json.dump(split, outfile)
    
                        

        


if __name__ == '__main__':
    main()