from cmath import nan
import numpy as np
import pandas as pd
import shutil

from dataUtil import DataUtil

'''
author Sakin Kirti
date 6/2/2022

script moves the patients that have a gleeson grade score to another folder
'''

# define the path containing all of the patients
datapath = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/clean'
destpath = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/ggg-confirmed'

dataspread = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/racial_disparity_FINAL_batch1_anonymized.xlsx'

def main():
    # open the anonymized batch file and isolate the patients that have gleeson grade score
    batch_anon = pd.read_excel(dataspread)
    batch_ggg_iso = batch_anon[batch_anon['GGG (1-5)'].notnull()]
    batch_ggg_names = batch_ggg_iso['ID'].values

    # iterate through names and copy+paste those files to another dir
    for name in batch_ggg_names:
        ID = f'prostate-{name.split(" ")[-1]}'

        # copy+paste
        try:
            shutil.copytree(f'{datapath}/{ID}', f'{destpath}/{ID}')
            print(f'copied and pasted {ID}')
        except FileNotFoundError:
            print(f'{ID} has no lesion marks')
        except FileExistsError:
            print(f'{ID} already exists')

if __name__ == '__main__':
    main()