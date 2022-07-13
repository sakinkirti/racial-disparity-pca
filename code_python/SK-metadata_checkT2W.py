import SimpleITK as sitk

from dataUtil import DataUtil as DU
from imageDataUtil import ImageDataUtil as IDU

'''
@author Sakin Kirti
@date 7/12/2022

script to print out the metadata for a T2W image
use to test if origin needs adjustment
'''

path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/AA_ggg-confirmed'
corr_origin = (0, 0, 0)
corr_spacing = (0.625, 0.625, 3.0)
corr_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

def main():
    # get the files to descend into
    patients = DU.getSubDirectories(path)

    for patient in patients:
        # get the T2W image
        T2W = f'{patient}/T2W.nii.gz'
        try: 
            image = sitk.ReadImage(T2W)
        except:
            image = sitk.ReadImage(f'{patient}/T2W_std.nii.gz')
        
        # get image metadata
        origin, spacing, direction = IDU.get_image_data(image)
        # print(f'patient: {str(patient).split("/")[-1]}, {corr_direction == direction}')

        # set the image metadata
        IDU.set_image_data(image, corr_origin, spacing, corr_direction)

        # double check and print
        origin, spacing, direction = IDU.get_image_data(image)
        print(f'patient: {str(patient).split("/")[-1]}, origin: {origin}, spacing: {spacing}, direction: {direction}')

        sitk.WriteImage(image, T2W)

if __name__ == '__main__':
    main()