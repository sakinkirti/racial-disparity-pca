from codecs import ignore_errors
import os
import glob
import SimpleITK as sitk
import shutil

from dataUtil import DataUtil as DU
from imageDataUtil import ImageData as IDU

UHRD_path = '/Users/sakinkirti/Downloads'
BCR_path = '/Users/sakinkirti/Downloads/drive-download-20220711T220137Z-001'
dst_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/RA_RDCasesFilteredAll'

paths = DU.getSubDirectories(dst_path)
for path in paths:
    under_path = DU.getSubDirectories(path)

    for under in under_path:
        shutil.rmtree(under, ignore_errors=True)