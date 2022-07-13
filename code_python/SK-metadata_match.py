from dataUtil import DataUtil
from imageDataUtil import ImageData

import os
import glob
import SimpleITK as sitk

'''
@author sakinkirti
@date 5/23/2022

script to replace all of the lesion mask metadata with the T2W image metadata
'''

main_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/RA_RDCasesFilteredAll'

def main(type):
    if type == 'single':
        img_path = glob.glob(f'{str(main_path)}/T2W.nii.gz', recursive=True)
        mask_paths = glob.glob(f'{str(main_path)}/T2W_LS.nii.gz', recursive=True)

        # get the image data
        img = sitk.ReadImage(img_path[0])
        origin, spacing, direction = ImageData.get_image_data(img)

        # iterate through the masks and set their metadata
        for mask_path in mask_paths:
            mask = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(mask)
            new_mask = sitk.GetImageFromArray(mask_arr)

            ImageData.set_image_data(new_mask, origin, spacing, direction)
            sitk.WriteImage(new_mask, mask_path)

        # double check
        print('double checking')

        # iterate through the masks and check their metadata
        for mask_path in mask_paths:
            mask = sitk.ReadImage(mask_path)
            mask_orig, mask_spacing, mask_direc = ImageData.get_image_data(mask)
                
            if origin != mask_orig:
                print(f'1 {mask_path} {origin} {mask_orig}')
            if spacing != mask_spacing:
                print(f'2 {mask_path} {spacing} {mask_spacing}')
            if direction != mask_direc:
                print(f'3 {mask_path} {direction} {mask_direc}')

    else:
        # get the subdirectories 
        subdirs = DataUtil.getSubDirectories(main_path)
        for dir in subdirs:
            # get the image and masks
            img_path = glob.glob(f'{str(dir)}/T2W.nii.gz', recursive=True)
            mask_paths = glob.glob(f'{str(dir)}/T2W_LS.nii.gz', recursive=True)

            # get the img metadata
            for image in img_path:
                img = sitk.ReadImage(image)
                origin, spacing, direction = ImageData.get_image_data(img)

                # iterate through the masks and set their metadata
                for mask_path in mask_paths:
                    mask = sitk.ReadImage(mask_path)
                    mask_arr = sitk.GetArrayFromImage(mask)
                    new_mask = sitk.GetImageFromArray(mask_arr)

                    ImageData.set_image_data(new_mask, origin, spacing, direction)
                    sitk.WriteImage(new_mask, mask_path)

        print('beginning double check')
        # double check that the metadata are the same
        for dir in subdirs:
            # get the image and masks
            img_path = glob.glob(f'{dir}/T2W.nii.gz', recursive=True)
            mask_paths = glob.glob(f'{dir}/LS*.nii.gz', recursive=True)

            # get the img metadata
            for image in img_path:
                img = sitk.ReadImage(image)
                origin, spacing, direction = ImageData.get_image_data(img)

                # iterate through the masks and set their metadata
                for mask_path in mask_paths:
                    mask = sitk.ReadImage(mask_path)
                    mask_orig, mask_spacing, mask_direc = ImageData.get_image_data(mask)
                    
                    if origin != mask_orig:
                        print(f'1 {mask_path} {origin} {mask_orig}')
                    if spacing != mask_spacing:
                        print(f'2 {mask_path} {spacing} {mask_spacing}')
                    if direction != mask_direc:
                        print(f'3 {mask_path} {direction} {mask_direc}')
        print('completed double check')

if __name__ == '__main__':
    main('many')