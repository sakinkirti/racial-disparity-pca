import glob

from dataUtil import DataUtil as DU

parent_path = '/Users/sakinkirti/Programming/Python/CCIPD/CA_clean'
pattern = 'ADC_reg.nii.gz'

def main():
    print('checking for files')

    patients = DU.getSubDirectories(parent_path)
    for patient_path in patients:
        file = glob.glob(pathname=f'{patient_path}/{pattern}', recursive=True)

        if len(file) == 0:
            print(patient_path)

    print('done')

if __name__ == '__main__':
    main()