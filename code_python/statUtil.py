import os
import numpy as np
import pandas as pd
from glob import glob
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import SimpleITK as sitk
from sklearn import metrics
from tqdm import tqdm

def DiceScore(input, target):
        smooth = 1.
        iflat = input.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()
        
        return ((2.*intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

class StatUtil(object):
    def __init__(self):
        pass
    
    @staticmethod
    def jaccard_index(rad1, rad2):
        '''
        Function to determine the similarity of two lists
        
        0.0 = no similarity
        1.0 = perfect similarity
        
        rad1: str indicating reader 1
        rad2: str indicating reader 2
        output: similarity score
        '''
        pats = os.listdir(fr'annotations_master/px1Label_{rad1}')
    
        all_jacc = []
        
        pbar = tqdm(pats)
        for pat in pbar:
            pbar.set_description(fr'Rad1: {rad1}, Rad2: {rad2}, Pat: {pat}')
            
            lsdirs1 = glob(fr'annotations_master/px1Label_{rad1}/{pat}/*.nii')
            lsdirs2 = glob(fr'annotations_master/px1Label_{rad2}/{pat}/*.nii')
            
            for i, _ in enumerate(lsdirs1):
                ls1 = sitk.GetArrayFromImage(sitk.ReadImage(lsdirs1[i])).astype('uint16')
                ls2 = sitk.GetArrayFromImage(sitk.ReadImage(lsdirs2[i])).astype('uint16')
                
                ls1[ls1>0] = 1
                ls2[ls2>0] = 1
                
                for slc in range(ls1.shape[0]):
                    if len(np.unique(ls1[slc])) > 1 and len(np.unique(ls2[slc])) > 1:
                        
                        try:
                            ls_jacc = metrics.jaccard_score(ls1[slc], ls2[slc], average='samples', zero_division=1)
                        except:
                            import pdb; pdb.set_trace()
                
                        all_jacc.append(ls_jacc)
        return sum(all_jacc)/len(all_jacc)
        
    @staticmethod
    def cohens_kappa(pred1, pred2):
        '''
        Function to determine the similarity of pairwise predictions
        
        0.0 = no similarity
        1.0 = perfect similarity
        
        list1: list/array of predictions 1
        list2: list/array of predictions 1
        output: kappa statistic
        '''
        kappa = metrics.cohen_kappa_score(pred1, pred2)
        
        return kappa
    
        
    @staticmethod
    def dice_coef(im1, im2):
        '''
        Function to determine the overlap of multiple binary images
        
        0.0 = no similarity
        1.0 = perfect similarity
        
        list1: list/array of predictions 1
        list2: list/array of predictions 1
        output: kappa statistic
        '''
        
        dicescore = []
        
        for slc in range(im1.shape[0]):
            if len(np.unique(im1[slc])) > 1 and len(np.unique(im2[slc])) > 1:
                dicescore.append(DiceScore(im1[slc], im2[slc]))
        
        avgdice = sum(dicescore)/len(dicescore)
        
        return avgdice
    