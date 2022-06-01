%{ 
author Sakin Kirti
date 5/29/2022

script to perform standardization of T2W and ADC images
%}

% gather the templates
templatePath = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/standardization';
%template_T2W = niftiread([templatePath filesep 'T2W_template.nii']);
template_ADC = niftiread([templatePath filesep 'ADC_template.nii']);
template_mask = niftiread([templatePath filesep 'PM_template.nii']);

%templateVolMasked_T2W = double(template_T2W) .* double(template_mask);
templateVolMasked_ADC = double(template_ADC) .* double(template_mask);
% opts.temcancermasks = logical(template_caMask);
opts.docheck = false;
opts.dorescale = false;

% get the list of patients and iterate
datapath = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/clean'; cd(datapath);
patients = split(ls); % {'prostate-00751'; ''}; 
for i = 1:size(patients,1)-1
    disp(['reading study - ' patients{i}]);
    
    % read the nifti files
    %T2W = niftiread([datapath filesep patients{i} filesep 'T2W.nii.gz']);
    ADC = niftiread([datapath filesep patients{i} filesep 'ADC_reg.nii.gz']);
    mask = niftiread([datapath filesep patients{i} filesep 'PM.nii.gz']);

    % store the metadata to be rewritten to the standardized files
    hdr = niftiinfo([datapath filesep patients{i} filesep 'T2W.nii.gz']);
    
    %T2WinputVolMask = double(T2W) .* double(mask);
    ADCinputVolMask = double(ADC) .* double(mask);
    cd(templatePath)
    %[~,T2WstdMap,~] = int_stdn_landmarks(T2WinputVolMask, templateVolMasked_T2W, opts);
    [~,ADCstdMap,~] = int_stdn_landmarks(ADCinputVolMask, templateVolMasked_ADC, opts);
    close all;
   
    % apply standardization
    %T2Wstd = applystdnmap_rs(T2W, T2WstdMap);
    ADCstd = applystdnmap_rs(ADC, ADCstdMap);
    
    % write the nifti image
    %niftiwrite(T2Wstd, [datapath filesep patients{i} filesep 'T2W_std.nii'], hdr, 'Compressed',true); 
    niftiwrite(ADCstd, [datapath filesep patients{i} filesep 'ADC_std.nii'], hdr, 'Compressed',true);
end

