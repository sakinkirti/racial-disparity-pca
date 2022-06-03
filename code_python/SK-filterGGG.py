from cmath import nan
import numpy as np
import pandas as pd

from dataUtil import DataUtil

'''
author Sakin Kirti
date 6/2/2022

script moves the patients that have a gleeson grade score to another folder
'''

# define the path containing all of the patients
datapath = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/clean'
dataspread = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/racial_disparity_FINAL_batch1_anonymized.xlsx'

def main():
    # open the anonymized batch file and isolate the patients that have gleeson grade score
    batch_anon = pd.read_excel(dataspread)
    print(batch_anon['GGG (1-5)'][1])
    batch_anon.loc[batch_anon['GGG (1-5)'] != nan]
    batch_ggg_names = batch_anon['ID'].values

    print(batch_ggg_names)

    # get the list of patients
    # patients = DataUtil.getSubDirectories(datapath)

if __name__ == '__main__':
    main()