import os
from dataUtil import DataUtil as du
import glob

'''
@author sakinkirti
@date 5/23/2022

a script to remove a common file from every folder in a path
'''

# the path holding all files
rm_path = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/to_nifti_cleaned'
format_to_remove = 'LS*_ADC_mask.*'

def main():
    # get each sub path
    folders = du.getSubDirectories(rm_path)
    print(f'removing files with format: {format_to_remove}')
    
    for f in folders:
        # get the files to remove form each folder
        rm_files = glob.glob(f'{f}/{format_to_remove}', recursive=True)

        # remove the files
        for file in rm_files:
            os.remove(file)
            print(f'removed file {file.split("/")[-1]}')

    print('finished removing files')

if __name__ == '__main__':
    main()