from genericpath import isdir
import glob
import os
import shutil

from dataUtil import DataUtil as DU

'''
author Sakin Kirti
date 6/5/2022

script to move lesion masks to their own separate folders based on the labeling group
'''

patient_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/CA_ggg-confirmed'
ls_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/CA_lesion-masks'
ls_groups = ['LB', 'ST']

race = 'CA'

def gather_AA_lesions():
    '''
    copy the AA lesions into a separate folder 
    '''

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

def gather_CA_lesions():
    '''
    copy the CA lesions into a separate folder 
    '''

    # descend to ggg-confirmed and get the masks for each lesion group
    patients = DU.getSubDirectories(patient_path)
    for patient in patients:
        T2W_lesions = glob.glob(f'{patient}/T2W_LS.nii.gz', recursive=True)
        ADC_lesions = glob.glob(f'{patient}/ADC_LS.nii.gz', recursive=True)
        LS_lesions = glob.glob(f'{patient}/LS1.nii.gz', recursive=True)

        # generate the required folder in ls_path
        dest_path = f'{ls_path}/{str(patient).split("/")[-1]}'
        if not os.path.isdir(dest_path):
            os.mkdir(dest_path)

        # copy the files to the desired folder
        for ls in T2W_lesions:
            shutil.copy(ls, f'{dest_path}/{str(ls).split("/")[-1]}')
        for ls in ADC_lesions:
            shutil.copy(ls, f'{dest_path}/{str(ls).split("/")[-1]}')
        for ls in LS_lesions:
            shutil.copy(ls, f'{dest_path}/{str(ls).split("/")[-1]}')

if __name__ == '__main__':
    if race == 'AA': gather_AA_lesions()
    elif race == 'CA': gather_CA_lesions()