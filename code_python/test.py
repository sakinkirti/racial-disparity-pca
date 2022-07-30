from codecs import ignore_errors
import os
import glob
import SimpleITK as sitk
import shutil
import pandas as pd

from dataUtil import DataUtil as DU
from imageDataUtil import ImageDataUtil as IDU

ca_csv_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/class_distribution_racial-disparity-pca_CA.csv'
patients = os.listdir('/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/CA_clean')

csv = pd.read_csv(ca_csv_path)
data = {}
for patient in patients:
    if patient in csv['PatiientID'].values:
        data[patient] = 'found'
    else:
        data[patient] = 'not found'

pd.DataFrame.from_dict(data=data, orient='index', columns=['status']).to_csv('/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/patient-status.csv')