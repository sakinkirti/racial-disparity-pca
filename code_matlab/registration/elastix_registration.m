%{
author Sakin Kirti
date 5/24/2022

script to run registration on ADC images based on T2W images using elastix
%}

% get the patient ids
destDir = '/Users/sakinkirti/Programming/Python/CCIPD/CA_clean';
cd(destDir)
patients = split(ls);

% store the elastix, transformix, and rigid pathnames
elastix = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/registration/elastix-5.0.0-mac/bin/elastix';
rigid = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/registration/rigid.txt';

% iterate through the patients
for i=1:size(patients,1)-1
    
    % if ADC_reg exists, then register it 
    ADC_reg = ['/Users/sakinkirti/Programming/Python/CCIPD/CA_clean/', patients{i}, '/ADC_reg.nii.gz'];
    if exist(ADC_reg, 'file') == false
        % isolate the ADC
        T2W = ['/', patients{i}, '/T2W.nii.gz'];

        % isolate ADC
        ADC = ['/', patients{i}, '/ADC.nii.gz'];
        
        % run elastix on ADC and T2
        system([elastix ' -f ' destDir T2W ' -m ' destDir ADC ' -p ' rigid ' -out ' destDir filesep patients{i}]);
        movefile([destDir filesep patients{i} filesep 'result.0.nii'], [destDir filesep patients{i} filesep 'ADC_reg.nii.gz']);
    end
end

% move back to the main dir at the end of the program
cd('/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab')
