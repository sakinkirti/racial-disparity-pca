from codecs import ignore_errors
import os
import glob
import SimpleITK as sitk
import shutil
import pandas as pd

from dataUtil import DataUtil as DU
from imageDataUtil import ImageDataUtil as IDU

path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/racial_disparity_FINAL_batch1_anonymized.xlsx'

df = pd.read_excel(path)

df = df[df['GGG (1-5)'].notnull()]
df.to_csv('/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/racial_disparity_ggg_filtered.csv')