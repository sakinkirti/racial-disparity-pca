import os
from dataUtil import DataUtil as du
import glob

'''
@author sakinkirti
@date 5/23/2022

a script to remove a common file from every folder in a path
'''

# the path holding all files
rm_path = '/Users/sakinkirti/Programming/Python/CCIPD/RA_clean'
format_to_remove = 'T2W_prostate-*_ADC_LS_overlay.jpg'

def main():
    # get each sub path
    folders = du.getSubDirectories(rm_path)
    print(f'removing files with format: {format_to_remove}')
    
    for f in folders:
        # get the files to remove from each folder
        rm_files = glob.glob(f'{f}/{format_to_remove}', recursive=True)

        # remove the files
        for file in rm_files:
            os.remove(file)
            filename = f'{file.split("/")[-2]}/{file.split("/")[-1]}'
            print(f'removed file {filename}')

    print('finished removing files')

if __name__ == '__main__':
    main()