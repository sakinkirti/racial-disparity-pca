import matlab.engine
import glob

from dataUtil import DataUtil as DU

'''
@author Sakin Kirti
@date 7/13/2022

script to call the matlab script to perform registration
'''

file_path = '/Volumes/GoogleDrive/.shortcut-targets-by-id/1UJRvU8BkLCs8ULNi-lIeGehkxcCSluw6/RacialDisparityPCa/data/RA_clean'

def main():
    # start the matlab engine
    print('starting MATLAB engine')
    engine = matlab.engine.start_matlab()

    # get patients
    patients = DU.getSubDirectories(file_path)
    for patient in patients:
        # isolate T2W and ADC images
        patient_path = str(patient)
        adc = glob.glob(pathname=f'{patient_path}/ADC.nii.gz', recursive=True)
        t2w = glob.glob(pathname=f'{patient_path}/T2W.nii.gz', recursive=True)
        if len(t2w) == 0: t2w = glob.glob(pathname=f'{patient_path}/T2W_std.nii.gz', recursive=True)

        # run registration only if there is an ADC and T2W image
        if len(adc) == 1 and len(t2w) == 1:
            engine.SingleRegistration(str(t2w[0]), str(adc[0]), patient_path)

    print('closing MATLAB engine')

if __name__ == '__main__':
    main()