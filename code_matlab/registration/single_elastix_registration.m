%{
author Sakin Kirti
date 5/31/2022

elastix registration for one image
%}

% the program filepaths
elastix = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/registration/elastix-5.0.0-mac/bin/elastix';
rigid = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/registration/rigid.txt';

% the image filepaths
dir = '/Users/sakinkirti/Downloads/drive-download-20220601T155025Z-001';
T2W = [dir filesep 'AH_T2W_template.nii'];
ADC = [dir filesep 'AH_ADC_template.nii'];

% run elastix on ADC and T2
system([elastix ' -f ' T2W ' -m ' ADC ' -p ' rigid ' -out ' dir])
movefile([dir filesep 'result.0.nii'], [dir filesep 'AH_ADC_template_reg.nii']);