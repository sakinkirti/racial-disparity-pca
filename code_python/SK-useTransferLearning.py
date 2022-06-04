import torch

from dataUtil import DataUtil as DU
from medImageProcessingUtil import MedImageProcessingUtil as MIPU
from augmentation3DUtil import Augmentation3DUtil as A3U

from AR_1patchwiseAugmentAndSaveAsHDF5 import getAugmentedData
from AR_4generatePredictions import SqueezeNet

'''
author Sakin Kirti
date 6/3/2022

script to use transfer learning from Ansh's SqueezeNet model
'''

model_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/AR-tl-models/checkpoint_SHT.pt'
hdf5_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-training/hdf5'

def main():
    # generate the hdf5 files on first run
    generate_hdf5(hdf5_path, do=True)

    # pass the model directly through Ansh's preloaded weights
    straight_through(model_path)

'''
method to generate hdf5 files, is always called
but if do is set to False nothing will happen
'''
def generate_hdf5(path, do=False):
    # generate only if do is true
    if do:
        getAugmentedData(path)
    else:
        return

'''
method to directly use one of the pretrained models
without training any of the layers
'''
def straight_through(path):
    weights = torch.load(path, map_location=torch.device('cpu'))

    model = SqueezeNet(weights)

if __name__ == '__main__':
    main()