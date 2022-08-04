import numpy as np
import pandas as pd
import glob
import os

aa_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-AA/racial_disparity_FINAL_batch1_anonymized.xlsx'
ca_path1 = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-CA/ClinicalInfoFinal_AS.xlsx'
ca_path2 = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-CA/SK_patientLabels_TP1_03092022.xlsx'
ca_path3 = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-CA/UH_BCR_clinical dat.xlsx'
ca_path4 = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-CA/AUA_RacialDisparity post Amr 10-30-2019.xlsx'
ca_path5 = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-CA/matched cohort_CCIPDmatch_anonymized_SHIVAM.xlsx'

ca_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/class_distribution_racial-disparity-pca_CA.csv'

output_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa'

def find_splits():
    '''
    get the spread of the data for AA and CA patients
    '''

    # create dataframe to add data to
    table = pd.DataFrame(columns=['race', 'num class 0', 'num class 1', 'total', 'proportion class 0', 'proportion class 1'])

    # read aa_path and quantify data
    aa_scores = pd.read_excel(aa_path)
    aa_scores = aa_scores[aa_scores['GGG (1-5)'].notnull()]

    # count the number of patients with each GGG score
    num1 = 0
    num0 = 0
    for index, row in aa_scores.iterrows():
        if row['GGG (1-5)'] > 1:
            num1 += 1
        else:
            num0 += 1

    # append to the table
    table.loc[len(table) + 1] = ['AA', num0, num1, num0+num1, num0/(num0+num1), num1/(num0+num1)]

    # reset counters
    num1 = 0
    num0 = 0

    # read ca_path1 and quantify data
    ca_scores = pd.read_csv(ca_path)
    ca_scores = ca_scores[ca_scores['GGG'].notnull()]
    for index, row in ca_scores.iterrows():
        if row['GGG'] > 1:
            num1 += 1
        else:
            num0 += 1
        
    # append to the table
    table.loc[len(table) + 1] = ['CA', num0, num1, num0+num1, num0/(num0+num1), num1/(num0+num1)]

    print(table)
    table.to_csv(path_or_buf=f'{output_path}/class_distribution_racial-disparity-pca.csv')

def generate_ggg():
    '''
    from the 4 ca_paths, generate 1 spreadsheet with all patients and ggg scores
    '''

    # store patients and ggg scoores {prostate-xxxxxx: y}
    dict = {}
    patient_list = os.listdir('/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/CA_clean')

    ca_scores = pd.read_excel(ca_path1)
    ca_scores = ca_scores[ca_scores['Initi GS1'].notnull()]
    for index, row in ca_scores.iterrows():
        if f'prostate-{row["ImgName"].split("_")[-1]}' not in dict.keys() and f'prostate-{row["ImgName"].split("_")[-1]}' in patient_list:
            dict[f'prostate-{row["ImgName"].split("_")[-1]}'] = row['Initi GS1']

    ca_scores = pd.read_excel(ca_path2)
    ca_scores = ca_scores[ca_scores['LS1-GGG'].notnull()]
    for index, row in ca_scores.iterrows():
        if f'prostate-{row["Patient ID"].split(" ")[-1]}' not in dict.keys() and f'prostate-{row["Patient ID"].split(" ")[-1]}' in patient_list:
            dict[f'prostate-{row["Patient ID"].split(" ")[-1]}'] = row['LS1-GGG']

    ca_scores = pd.read_excel(ca_path3)
    ca_scores = ca_scores[ca_scores['bGG'].notnull()]
    for index, row in ca_scores.iterrows():
        if f'prostate-{str(row["pID"]).zfill(6)}' not in dict.keys() and f'prostate-{str(row["pID"]).zfill(6)}' in patient_list:
            dict[f'prostate-{str(row["pID"]).zfill(6)}'] = row['bGG']

    ca_scores = pd.read_excel(ca_path4)
    ca_scores = ca_scores[ca_scores['GGG'].notnull()]
    for index, row in ca_scores.iterrows():
        if f'prostate-{str(row["patientID"]).split(" ")[-1]}' not in dict.keys() and f'prostate-{str(row["patientID"]).split(" ")[-1]}' in patient_list:
            dict[f'prostate-{str(row["patientID"]).split(" ")[-1]}'] = row['GGG']

    ca_scores = pd.read_excel(ca_path5)
    ca_scores = ca_scores[ca_scores['GGG'].notnull()]
    for index, row in ca_scores.iterrows():
        if f'prostate-{str(row["ProstateID"]).split(" ")[-1]}' not in dict.keys() and f'prostate-{str(row["ProstateID"]).split(" ")[-1]}' in patient_list:
            dict[f'prostate-{str(row["ProstateID"]).split(" ")[-1]}'] = row['GGG'] if row['GGG'] != 'Negative' else 1

    table = pd.DataFrame.from_dict(dict, orient='index', columns=['GGG'])
    table.to_csv(path_or_buf=f'{output_path}/class_distribution_racial-disparity-pca_CA.csv')
    print(table)
    import pdb; pdb.set_trace()

if __name__ == '__find_splits__':
    generate_ggg()
    find_splits()