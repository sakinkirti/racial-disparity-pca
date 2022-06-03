import os
import glob
import pandas as pd

from dataUtil import DataUtil

'''
author Sakin Kirti
date 6/3/2022

script to collect the patients that have more than one lesion (patients with LS2)
'''

datapath = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/ggg-confirmed'
savepath = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa'
lesions = []

def main():
    # get the list of patients and iterate
    patients = DataUtil.getSubDirectories(datapath)
    for patient in patients:
        excess_ls = glob.glob(f'{patient}/LS*', recursive=True)

        # remove any LS1 from list
        temp_holder = []
        for ls in excess_ls:
            lesion = ls.split('/')[-1]
            print(list(lesion.split('_')[0])[-1])
            if list(lesion.split('_')[0])[-1] != '1':
                temp_holder.append(lesion)

        # insert the patient id to the list of lesions and append to the lesions array
        temp_holder.insert(0, str(patient).split('/')[-1])
        lesions.append(temp_holder)
        #print(f'appended {temp_holder}')

    # save the array as a pandas dataframe and export as csv
    extra_lesions_df = pd.DataFrame(lesions, columns=['patient id', 'LS', 'LS'])
    pd.DataFrame.to_csv(extra_lesions_df, f'{savepath}/extra_lesions_rdpca.csv')
    print('saved df')

if __name__ == '__main__':
    main()