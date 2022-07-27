# import necessary packages
from socketserver import DatagramRequestHandler
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np 
from skimage import measure
from skimage.measure import regionprops
from sympy import N
from dataUtil import DataUtil as DU
import glob
from imageDataUtil import ImageDataUtil as IDU

# store the paths to all nifti images
nifti_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/RA_clean/prostate-000875/ADC.nii.gz'
save_path = '/Users/sakinkirti/Desktop/test/to_nifti_cleaned'

num_images = 1 # 1 means just 1 image, any other number will mean many images

def main(num_images):
    if num_images != 1:
        # get the paths
        paths = DU.getSubDirectories(nifti_path)
        n = 0

        # iterate through each path
        for path in paths:
            # get the individual t2w nifti files
            imgs = glob.glob(str(path) + '/T2W*.nii.gz', recursive=False)

            # loop through the images
            for img_path in imgs:
                # load the image
                img = sitk.ReadImage(img_path)
                img_arr = sitk.GetArrayFromImage(img)

                # get and store the image metadata
                origin, spacing, direction = IDU.get_image_data(img)

                # flip the image array
                img_arr = np.flip(img_arr, 0)

                # store the image array as an sitk file and set the image data
                flipped = sitk.GetImageFromArray(img_arr)
                IDU.set_image_data(flipped, origin, spacing, direction)

                # set the filename
                filename = save_path + '/' + str(path).split('/')[-1] + '/' + str(img_path).split('/')[-1].split('.')[0] + '.nii.gz'

                # write the file
                writer = sitk.ImageFileWriter()
                writer.SetFileName(filename)
                writer.Execute(flipped)
                print('completed ' + str(path).split('/')[-1] + '/' + str(img_path).split('/')[-1].split('.')[0])
                n += 1
    else:
        # load the image
        img = sitk.ReadImage(nifti_path)
        img_arr = sitk.GetArrayFromImage(img)

        # get and store the image metadata
        origin, spacing, direction = IDU.get_image_data(img)

        # flip the image array
        img_arr = np.flip(img_arr, 0)

        # store the image array as an sitk file and set the image data
        flipped = sitk.GetImageFromArray(img_arr)
        IDU.set_image_data(flipped, origin, spacing, direction)

        # set the filename
        filename = nifti_path + '_flipped.nii.gz'

        # write the file
        writer = sitk.ImageFileWriter()
        writer.SetFileName(filename)
        writer.Execute(flipped)
        print('completed 1 image')

    print('done')

if __name__ == '__main__':
    main(num_images)