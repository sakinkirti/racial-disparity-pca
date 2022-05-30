%{
author Sakin Kirti
date 5/24/2022

script to run registration on ADC images based on T2W images using elastix
%}

% get the patient ids
destDir = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/to_nifti_cleaned';
cd(destDir)
patients = split(ls);

% store the elastix, transformix, and rigid pathnames
elastix = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/elastix-5.0.0-mac/bin/elastix';
rigid = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/rigid.txt';

% iterate through the patients
for i=1:size(patients,1)-1
    % isolate the T2 and ADC images
    T2W = ['/', patients{i}, '/T2W.nii.gz'];
    ADC = ['/', patients{i}, '/ADC.nii.gz'];
    
    % run elastix on ADC and T2
    system([elastix ' -f ' destDir T2W ' -m ' destDir ADC ' -p ' rigid ' -out ' destDir filesep patients{i}])
    movefile([destDir filesep patients{i} filesep 'result.0.nii.gz'], [destDir filesep patients{i} filesep 'ADC_reg.nii.gz']);
end

% move back to the main dir at the end of the program
cd('/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab')