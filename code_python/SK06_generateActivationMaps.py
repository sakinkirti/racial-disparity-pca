# import packages
import shutup; shutup.please(); 
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
import skimage.measure

from pathlib import Path
from torchvision import models
from torch.autograd import Variable
from skimage.transform import resize as skresize

from SK00_racialDisparitySqueezeNet import SqueezeNet

'''
@author Sakin Kirti
@date 6/24/2022

script to generate activation maps for the model created
An activation map shows the region of the image used by the CNN to make its prediction
'''

# set the os environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# global variables used throughout script
source_dir = os.getcwd() # str(Path(os.getcwd()).parent)
test_set_dir = f'{source_dir}/model-outputs/racial-disparity-hdf5/test.h5'
model_storage = f'{source_dir}/model-outputs/racial-disparity-models'

save_path = f'{source_dir}/model-outputs/rdp-activation-maps'
img_out_size = (224, 224)

rads = ['ST', 'LB']

'''
method to load the data
'''
def load_data(set_dir):
    print('\nLOADING DATA...')

    # load in the file
    file = h5py.File(set_dir, mode='r', libver='latest')

    # separate the data from the labels
    data = file['data']
    labels = file['labels']
    names = file['names']

    # return and print
    print('done')
    return data, labels, names

'''
method to load in pre-trained model weights to squeezenet architecture
'''
def get_model(checkpoint):
    print('\nGENERATING A SQUEEZENET ARCHITECTURE AND LOADING IN THE WEIGHTS...')

    # set up the model from torchvision.models
    model = SqueezeNet(checkpoint_path=checkpoint)

    # set the model to evaluate
    model.eval()

    # return and print
    print('done')
    return model

'''
method to generate heatmaps
'''
def generate_heatmap(model, img, cls):
    # clear the image and load new
    img = img[None]
    img = torch.from_numpy(img)

    # run the img through the model
    pred = model(img)
    probs = F.softmax(pred, dim=1)
    pred[:, cls].backward()
    gradients = model.get_activations_gradient()

    # compute the mean pooling for the calculated gradients
    pooled_gradients = torch.mean(gradients, dim = 0); pooled_gradients = torch.mean(pooled_gradients, dim = 1); pooled_gradients = torch.mean(pooled_gradients, dim = 1);

    # isolate the activations
    activations = model.get_activations(img).detach()
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # create the heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.detach().data.cpu().numpy()
    heatmap = np.nan_to_num(heatmap)

    # finalize the probs
    probs = probs.detach().data.cpu().numpy().ravel()[-1]

    # return
    return heatmap, probs

'''
method to overlay the mask over the image
'''
def overlay_mask(img, mask):
    # get the outline of the mask
    contours = skimage.measure.find_contours(mask, 0)

    # set the plot figure size
    plt.figure(figsize=(5,5))

    # show the image
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)

    # overlay the mask outline
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color='g')

    # no axes
    plt.xticks([])
    plt.yticks([])

    # return
    return plt

'''
method to overlay the heatmap over top of the mask-overlayed image
'''
def overlay_heatmaps(img, map_1, map_2, mask):
    # get the outline of mask
    contours = skimage.measure.find_contours(mask, 0)

    # set the figure size
    plt.figure(figsize=(5,5))

    # show the image
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)

    # overlay the heatmaps over the image
    plt.imshow(map_2, cmap='Reds', interpolation=None, vmin=0, vmax=1, alpha=map_2)
    plt.imshow(map_1, cmap='Blues', interpolation=None, vmin=0, vmax=1, alpha=map_1)

    # overlay mask outline
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color='g')

    # no axes
    plt.xticks([])
    plt.yticks([])

    # return
    return plt 

'''
method loops over data and calls methods to...
- generate heatmaps
- overlap masks
- overlay heatmaps
'''
def generate_save_heatmaps(model, save_path, data, labels, names, rad):
    print(f'\nGENERATING AND SAVING HEATMAPS FOR {rad}...')

    # iterate through the data
    for j,sample in enumerate(range(data.shape[0])):
        heatmaps = []

        # get the name and label and print for user's progress
        name = names[sample].decode('utf-8')
        label = labels[sample]
        print(f'Name: {name}, Label: {label}, Progress: {j / data.shape[0]}')

        # load the images
        orig_img = data[sample,:,:]
        img = data[sample]
        slices = np.unique(np.nonzero(img)[0])

        # iterate through the output class
        for cls in [0, 1]:
            # generate the heatmap
            heatmap, probs = generate_heatmap(model, img, cls)

            # save only if it is the correct output, otherwise check the other label
            if (label == 0 and probs > 0.0 and probs < 0.4) or (label == 1 and probs > 0.6 and probs < 1.0):
                pass

            # set the output folder and create if it doesnt exist
            out_path = f'{save_path}/{rad}/label-{label}'
            if not os.path.exists(out_path): os.makedirs(out_path)

            # resize the heatmap and add it to the list of heatmaps
            heatmaps.append(skresize(heatmap, output_shape=img_out_size, order=1))
        
        # if there are multiple heatmaps
        if len(heatmaps) > 0:
                mask = orig_img[2]

                if mask.ndim != 2:
                    import pdb 
                    pdb.set_trace()

                orgimg = orig_img[0]

                plt_org = overlay_mask(orgimg, mask)
                # plt_org.show()
                plt_org.savefig(f'{out_path}/{name}_({label},{probs})_org_{cls}.jpg', bbox_inches='tight', pad_inches=0)
                plt.close()

                plt_heat = overlay_heatmaps(orgimg, heatmaps[0], heatmaps[1], mask)
                # plt_heat.show()
                plt_heat.savefig(f'{out_path}/{name}_({label},{probs})_map_{cls}.jpg', bbox_inches='tight', pad_inches=0)
                plt.close()

    # print
    print('done')

'''
main method
'''
def main():
    print('ACTIVATION MAPS UPCOMING...')

    # load the data
    data, labels, names = load_data(test_set_dir)

    for rad in rads:
        # generate model and load in the weights
        model = get_model(checkpoint=f'{model_storage}/rdp-aa-{rad}/early-stop_{rad}.pt')

        # generate and save heatmaps
        generate_save_heatmaps(model=model, save_path=save_path, data=data, labels=labels, names=names, rad=rad)

if __name__ == '__main__':
    main()