function x = SingleRegistration(T2W, ADC, patient_path)
% author: Sakin Kirti
% date: 5/31/2022
%
% single_registration - a function to run elastix registration on a single image

    % the program filepaths
    elastix = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/registration/elastix-5.0.0-mac/bin/elastix';
    rigid = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/registration/rigid.txt';

    % run elastix on ADC with T2W
    system([elastix ' -f ' T2W ' -m ' ADC ' -p ' rigid ' -out ' patient_path])

    % rename result to ADC_reg
    movefile([patient_path filesep 'result.0.nii'], [patient_path filesep 'ADC_reg.nii.gz']);
    x = 1
    
end
