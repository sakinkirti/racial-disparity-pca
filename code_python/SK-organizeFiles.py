import os
import glob
import shutil

from sqlalchemy import true

path = '/Users/sakinkirti/Downloads/cases'
dest = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/RA_RDCasesFilteredAll'

def main():
    print('organizing files')

    # iterate through patient numbers
    for i in range (0, 500):
        # collect files to move
        file_counter = f'{i:03}'
        files = glob.glob(pathname=f'{path}/*{file_counter}*', recursive=true)

        # if files are collected, then move to new folder
        if len(files) > 0:
            # make the directory for storage
            new_path = f'{dest}/prostate-000{file_counter}'
            os.mkdir(path=new_path)

            # move the files
            for file in files:
                shutil.move(src=file, dst=f'{new_path}/{file.split("/")[-1]}')
    
if __name__ == '__main__':
    main()