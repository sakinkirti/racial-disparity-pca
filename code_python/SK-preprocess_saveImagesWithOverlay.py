import sys
from pathlib import Path
sys.path.append(str(Path(f"../Code_general")))
from dataUtil import DataUtil 
import SimpleITK as sitk 
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import glob 

'''
@author: sakinkirti
@date: 04/12/2022

This file takes a folder that contains folders that have T2W and lesion masks as nifti files and overlays the masks over the images
'''

# define the path to the nifti images
path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/RA_clean'
save_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/RA_mask-overlaid'

'''
Main method, takes the path that holds all of the folders in which the images and masks are held
Overlays the masks on top of the image
'''
def main(path_to_niftis):
    # get a list of folders in path_to_niftis
    dirs = DataUtil.getSubDirectories(path_to_niftis)

    # each dir contains both the nifti file and the lesion mask - lesions masks names LS_.nii.gz - imgs titled T2W.nii.gz
    for dir in dirs:
        # get the img paths
        img_pths = glob.glob(str(dir) + '/T2W.nii.gz', recursive=True)
        if len(img_pths) == 0: img_pths = glob.glob(str(dir) + '/T2W_std.nii.gz', recursive=True)
        # get the mask path
        msk_pth = glob.glob(str(dir) + '/T2W_LS.nii.gz', recursive=True)
        if len(msk_pth) == 0: msk_pth = glob.glob(str(dir) + '/LS1.nii.gz', recursive=True)

        # loop through the masks if there are multiple of each - there is only one image so no looping is needed for images
        for msk in msk_pth:
            # read the images
            for img_pth in img_pths:
                img = sitk.ReadImage(img_pth)
                lesion = sitk.ReadImage(msk)

                # convert to np array
                img_np = sitk.GetArrayFromImage(img) # np.flip(sitk.GetArrayFromImage(img), 0)
                lesion_np = sitk.GetArrayFromImage(lesion) 

                # find the section with the marking
                slice = find_mask(lesion_np)
                if slice > img_np.shape[0]: slice = img_np.shape[0] - 1

                # save the image
                overlayedImg = overlay_mask(img_np[slice], [lesion_np[slice]])
                #plt.show()
                imgID = 'T2W_' + str(dir).split('/')[-1] + '_' + str(msk).split("/")[-1].split(".")[0]
                plt.savefig(f'{save_path}/{imgID}_overlay.jpg', bbox_inches='tight', pad_inches=0)
                plt.savefig(f'{str(dir)}/{imgID}_overlay.jpg', bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f'saved overlay for {imgID}')

'''
takes the lesion mask
returns the middle slice of the mask
'''
def find_mask(mask):
    # iterate through z sections to find the section with lesion markings
    slice, y_ind, x_ind = np.where(mask != 0)
    return slice[0]

'''
takes the img and the mask and overlays the mask
returns a figure containing the mask over the image
'''
def overlay_mask(orgimg,masks,colors=None):
    # plot the image
    plt.figure(figsize=(5,5))
    plt.imshow(orgimg, cmap = 'gray')

    # overlay the mask contour over the image
    if colors is None:
      colors = ['yellow']

    for i,mask in enumerate(masks): 
      color = colors[i]
      contours = measure.find_contours(mask, 0)
      for n, contour in enumerate(contours):
          plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color=color)

    plt.xticks([])
    plt.yticks([])

    return plt

# run the main method
if __name__ == '__main__':
    main(path) 