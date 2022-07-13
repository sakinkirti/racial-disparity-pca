import glob
import os
import shutil

from dataUtil import DataUtil as DU

'''
author Sakin Kirti
date 6/5/2022

script to move lesion masks to their own separate folders based on the labeling group
'''

patient_path = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/ggg-confirmed'
ls_path = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/lesion-masks'
ls_groups = ['LB', 'ST']

def main():
    # descend to ggg-confirmed and get the masks for each lesion group
    patients = DU.getSubDirectories(patient_path)
    for patient in patients:
        LB_lesions = glob.glob(f'{patient}/LS*_LB.nii.gz', recursive=True)
        ST_lesions = glob.glob(f'{patient}/LS*_ST.nii.gz', recursive=True)

        # iterate through each group and make the required folder
        for group in ls_groups:
            dest_path = f'{ls_path}/{group}/{str(patient).split("/")[-1]}'
            os.mkdir(dest_path)

            # copy the files to the desired folder
            for ls in LB_lesions:
                shutil.copy(ls, f'{dest_path}/{str(ls).split("/")[-1].split("_")[0]}.nii.gz')
            for ls in ST_lesions:
                shutil.copy(ls, f'{dest_path}/{str(ls).split("/")[-1].split("_")[0]}.nii.gz')

if __name__ == '__main__':
    main()