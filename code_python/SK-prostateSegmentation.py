from pathlib import Path
import glob
import subprocess

from dataUtil import DataUtil

'''
@author Sakin Kirti
@date 5\25\22

script to segment prostate gland from respective T2W image
NOTE: can only be run on Windows machines because proSeg only works on Windows

dependencies:
- Windows machine - running via Bootcamp or Parallels will work as well
- need the proSeg folder which contains packages for proSeg to run and proSeg.exe script
'''

data_path = r'C:\Users\Public\prostate_segmentation\data'
proseg_path = r'C:\Users\Public\prostate_segmentation\proSeg\proSeg.exe'

def main():
    # get list of patients and iterate through them
    patients = DataUtil.getSubDirectories(data_path)
    for patient in patients:
        # get the T2W image and set the output path
        T2W = glob.glob(str(patient) + r'\T2W.nii.gz', recursive=True)[0]
        PM = str(patient) + r'\PM.nii.gz'
        CG = str(patient) + r'\CG.nii.gz'

        # execute proSeg
        subprocess.run([proseg_path, T2W, PM, CG])

if __name__ == '__main__':
    main()