import glob
import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt

from dataUtil import DataUtil
from imageData import ImageData

# the path to the masks
mask_path = '/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/to_nifti_cleaned'
save_path = '/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/to_nifti_cleaned'

'''
main method
'''
def main(pathToMasks):

    print("starting")
    # iterate through paths within the main path
    paths = DataUtil.getSubDirectories(pathToMasks)
    for path in paths:

        # loop through each file
        masks = glob.glob(str(path) + '/LS*.nii.gz', recursive=False)
        for mask in masks:

            # get the image and its metadata
            mask_img = sitk.ReadImage(mask)
            mask_origin, mask_spacing, mask_direction = ImageData.get_image_data(mask_img)

            # store the np array of the image
            mask_np = sitk.GetArrayFromImage(mask_img)

            # start a new image
            new_mask = np.zeros(mask_np.shape, dtype=mask_np.dtype)

            # loop through the layers of mask_np
            for layer_num in range(0, mask_np.shape[0]-1):
                new_mask[layer_num,:,:] = mask_np[layer_num+1,:,:]
                plt.imshow(new_mask[layer_num-1,:,:])

            # add a new slice to the bottom of the array
            new_mask[mask_np.shape[0]-1,:,:] = mask_np[0,:,:]

            # set the metadata for the new image
            new_mask_img = sitk.GetImageFromArray(new_mask)
            ImageData.set_image_data(new_mask_img, mask_origin, mask_spacing, mask_direction)

            # filename
            filename = save_path + '/' + str(path).split('/')[-1] + '/' + str(mask).split('/')[-1]

            # save the image
            img_writer = sitk.ImageFileWriter()
            img_writer.SetFileName(filename)
            img_writer.Execute(new_mask_img)
            print("completed image " + filename)
    
'''
call the main method when run
'''
if __name__ == '__main__':
    main(mask_path)
