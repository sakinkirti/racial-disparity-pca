import json
import random
import pandas as pd

from sklearn.model_selection import train_test_split

'''
@author Sakin Kirti
@date 6/11/22

script to generate a json file that labels each training sample as train, val, test
'''

# path to spreadsheets containing GGG scores
aa_csv_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-AA/racial_disparity_FINAL_batch1_anonymized.xlsx'
ca_csv_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-CA/racial_disparity_ggg_filtered.csv'

# where to save the json splits
aa_json_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-AA/rdp-splits-json/racial-disparity-splits.json'
ca_json_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-CA/rdp-splits-json/racial-disparity-splits.json'
ra_json_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/model-outputs-RA/rdp-splits-json/racial-disparity-splits.json'

# the specifc race to generate splits for (AA, CA, or RA)
race = 'RA'

# random seed and the proportion to split the data into
rs = random.randint(1,1000)
split_prop = {'train' : 0.7, 
                'val' : 0.1, 
                'test' : 0.2}

def generate_splits_AA():
    '''
    method to generate splits for the specifically african american patients
    uses the aa_csv_path'''

    # read the csv file
    df = pd.read_excel(aa_csv_path)
    df = df[df['GGG (1-5)'].notnull()]
    
    sigs = {}
    for index, row in df.iterrows():
        patnum = row['ID'].split(' ')[-1]

        # split the data according to GGG
        sigs[f'prostate-{patnum}'] = 1 if row['GGG (1-5)'] > 1 else 0

    # conver to df
    sigs_df = pd.DataFrame(sigs.items(), columns=['ID', 'Sig'])

    # train test split
    train, val = train_test_split(sigs_df, test_size=0.51, random_state=rs)
    val, test = train_test_split(val, test_size=0.59, random_state=rs)

    # let the user know what the splits look like
    print(f'this is a {train.shape[0] / df.shape[0]} / {val.shape[0] / df.shape[0]} / {test.shape[0] / df.shape[0]} for train / val / test splits')
    print(f'train set [n: {train.shape[0]}, class 0%: {train["Sig"].value_counts()[0] / train.shape[0]}, class 1%: {train["Sig"].value_counts()[1] / train.shape[0]}]')
    print(f'val set [n: {val.shape[0]}, class 0%: {val["Sig"].value_counts()[0] / val.shape[0]}, class 1%: {val["Sig"].value_counts()[1] / val.shape[0]}]')
    print(f'test set [n: {test.shape[0]}, class 0%: {test["Sig"].value_counts()[0] / test.shape[0]}, class 1%: {test["Sig"].value_counts()[1] / test.shape[0]}]')
    
    split = {}
    for index, row in train.iterrows():
        split[row['ID']] = "train"
    for index, row in val.iterrows():
        split[row['ID']] = "val"
    for index, row in test.iterrows():
        split[row['ID']] = "test"
        
    with open(aa_json_path, 'w') as outfile:
        json.dump(split, outfile)

def generate_splits_CA():
    '''
    generate splits for the caucaisan american patients
    '''

    # read the csv
    df = pd.read_csv(ca_csv_path)
    df = df[df['GGG'].notnull()]
    
    sigs = {}
    for index, row in df.iterrows():
        patnum = row['PatientID'].split('-')[-1]

        # split the data according to GGG
        sigs[f'prostate-{patnum}'] = 1 if row['GGG'] > 1 else 0

    # convert to df
    sigs_df = pd.DataFrame(sigs.items(), columns=['ID', 'Sig'])

    # train test split
    train, val = train_test_split(sigs_df, test_size=0.3, random_state=rs)
    val, test = train_test_split(val, test_size=0.66, random_state=rs)

    # let the user know what the splits look like
    print(f'this is a {train.shape[0] / df.shape[0]} / {val.shape[0] / df.shape[0]} / {test.shape[0] / df.shape[0]} for train / val / test splits')
    print(f'train set [n: {train.shape[0]}, class 0%: {train["Sig"].value_counts()[0] / train.shape[0]}, class 1%: {train["Sig"].value_counts()[1] / train.shape[0]}]')
    print(f'val set [n: {val.shape[0]}, class 0%: {val["Sig"].value_counts()[0] / val.shape[0]}, class 1%: {val["Sig"].value_counts()[1] / val.shape[0]}]')
    print(f'test set [n: {test.shape[0]}, class 0%: {test["Sig"].value_counts()[0] / test.shape[0]}, class 1%: {test["Sig"].value_counts()[1] / test.shape[0]}]')
    
    split = {}
    for index, row in train.iterrows():
        split[row['ID']] = "train"
    for index, row in val.iterrows():
        split[row['ID']] = "val"
    for index, row in test.iterrows():
        split[row['ID']] = "test"
        
    with open(ca_json_path, 'w') as outfile:
        json.dump(split, outfile)

def generate_splits_RA():
    '''
    method to generate splits for a race agnostic model
    ie taking both african american and caucasian american patients
    similar method as the CA where each spreadsheet is added individually
    '''
    
    sigs = {}

    # ----- FIRST DEAL WITH THE AA PATIENTS ----- #
    # read the AA csv file
    df = pd.read_excel(aa_csv_path)
    df = df[df['GGG (1-5)'].notnull()]

    for index, row in df.iterrows():
        patnum = row['ID'].split(' ')[-1]

        # split the data according to GGG
        sigs[f'prostate-{patnum}'] = 1 if row['GGG (1-5)'] > 1 else 0

    # ----- THEN DEAL WITH THE CA PATIENTS ----- #
    # read the CA csv file
    df = pd.read_csv(ca_csv_path)
    df = df[df['GGG'].notnull()]
    
    for index, row in df.iterrows():
        patnum = row['PatientID'].split('-')[-1]

        # split the data according to GGG
        sigs[f'prostate-{patnum}'] = 1 if row['GGG'] > 1 else 0

    # convert to df
    sigs_df = pd.DataFrame(sigs.items(), columns=['ID', 'Sig'])

    # train test split
    train, val = train_test_split(sigs_df, test_size=0.3, random_state=rs)
    val, test = train_test_split(val, test_size=0.66, random_state=rs)

    # let the user know what the splits look like
    print(f'this is a {train.shape[0] / sigs_df.shape[0]} / {val.shape[0] / sigs_df.shape[0]} / {test.shape[0] / sigs_df.shape[0]} for train / val / test splits')
    print(f'train set [n: {train.shape[0]}, class 0%: {train["Sig"].value_counts()[0] / train.shape[0]}, class 1%: {train["Sig"].value_counts()[1] / train.shape[0]}]')
    print(f'val set [n: {val.shape[0]}, class 0%: {val["Sig"].value_counts()[0] / val.shape[0]}, class 1%: {val["Sig"].value_counts()[1] / val.shape[0]}]')
    print(f'test set [n: {test.shape[0]}, class 0%: {test["Sig"].value_counts()[0] / test.shape[0]}, class 1%: {test["Sig"].value_counts()[1] / test.shape[0]}]')
    
    split = {}
    for index, row in train.iterrows():
        split[row['ID']] = "train"
    for index, row in val.iterrows():
        split[row['ID']] = "val"
    for index, row in test.iterrows():
        split[row['ID']] = "test"
        
    with open(ra_json_path, 'w') as outfile:
        json.dump(split, outfile)

def main(race):
    '''
    differentiates between which split should be created and call that specific method
    '''

    if race == 'AA':
        generate_splits_AA()
    elif race == 'CA':
        generate_splits_CA()
    elif race == 'RA':
        generate_splits_RA()
    
if __name__ == '__main__':
    main(race)