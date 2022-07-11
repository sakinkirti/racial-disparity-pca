import os
import glob

from sqlalchemy import false, true

'''
@author Sakin Kirti
@since 6/7/2022

script to rename and organize files into the proper ones for preprocessing and model training
'''

# global vars
file_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/RA_RDCasesFilteredAll'
old_file_name = '*PIRADS*.nii.gz'
new_file_name = 'T2W_LS'
zipped = true

def main():
    print('renaming files')

    # descend into the filepath to get the individual patient folders
    patients = os.listdir(file_path)
    
    # descend into each patient and gather the files to rename
    for patient in patients:
        print(f'renaming files for patient: {patient}')
        patient_path = f'{file_path}/{patient}'

        # gathering the necessary file
        file = glob.glob(f'{patient_path}/{old_file_name}', recursive=true)

        if len(file) == 1:
            file = file[0]
            # check if zipped and rename and zip, otherwise just rename and save unzipped
            if zipped:
                os.rename(file, f'{patient_path}/{new_file_name}.nii.gz')
            else:
                os.rename(file, f'{patient_path}/T2W.nii')



if __name__ == '__main__':
    main()