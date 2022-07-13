from medImageProcessingUtil import MedImageProcessingUtil
import SimpleITK as sitk 
from glob import glob 
import numpy as np 
import pandas as pd 
import os 
import sys 
import yaml 
from progressbar import *  
import json 
import pickle
from pathlib import Path 
import matplotlib.pyplot as plt 

'''
define a class that can get image data from sitk images
'''
class ImageDataUtil(object):

    # initialize the object
    def init(self):
        pass

    '''
    method to get all of the image metadata
    @param the image the get data about
    @return orig - the origin of the image
    @return spacing - the spacing between slices
    @return direction - the direction of the image
    '''
    @staticmethod
    def get_image_data(img):
        # get the necessary data to reconstruct the image
        orig = img.GetOrigin()
        spacing = img.GetSpacing()
        direction = img.GetDirection()

        return orig, spacing, direction

    '''
    method to set the meta data to an image
    @param img - the image to set the data to
    @param origin - the origin of the image
    @param spacing - the spacing between slices
    @param direction - the direction of the image
    '''
    @staticmethod
    def set_image_data(img, origin, spacing, direction):
        # set the data
        img.SetOrigin(origin)
        img.SetSpacing(spacing)
        img.SetDirection(direction)