%{ 
author Pranav Kirti
date 5/30/2022

script to perform standardization of T2W and ADC images
%}

% gather the templates
templatePath = 'G:\.shortcut-targets-by-id\1CxPvrtnpodCQ7_BNcO6dTfs1xhZNu3-kJ\Template';
template_T2 = niftiread([templatePath filesep 'T2W.nii']);
template_ADC = niftiread([templatePath filesep 'ADC.nii']);
template_mask = niftiread([templatePath filesep 'PM.nii']);

% template_caMask = mha_read_volume([templatePath filesep 'T2-label.mha']); template_caMask = logical(template_caMask);
% template_mask = imdilate(template_caMask,strel('disk',18));

templateVolMasked_T2 = double(template_T2) .* double(template_mask);
% templateVolMasked_ADC = double(template_ADC).*template_mask;
% opts.temcancermasks = logical(template_caMask);
opts.docheck = false;
opts.dorescale = false;

% get the list of patients and iterate
datapath = '/Users/sakinkirti/Programming/Python/CCFIPD/racial-disparity-pca/dataset/clean'; cd(datapath);
patients = split(ls);
for i = 1:length(patients,1)-1
    disp(['reading study - ' patients{i}]);
    
    % read the nifti images
    T2W = niftiread([datapath filesep patients{i} filesep 'T2W.nii.gz']);
    ADC = niftiread([datapath filesep patients{i} filesep 'ADC_reg.nii.gz']);
    mask = niftiread([datapath filesep patients{i} filesep 'LS*.nii.gz']);
    
    inputVolMask = double(T2W) .* double(mask);
    [~,stdMap,~] = int_stnd_landmarks(inputVolMask,templateVolMasked_T2,opts);
    close all;
   
    
    % apply standardization
    T2Wstd = applystdnmap_r(T2, stdMap);
    
    % write the nifti image
    niftiwrite(T2Wstd, [datapath filesep patients{i} filesep 'T2W_std.nii.gz'], ); 
    niftiwrite(ADCstd, [datapath filesep patients{i} filesep 'ADC_std.nii.gz'], );
    ([datapath filesep pats{i}(13:end) '_T2_std.mha'],T2std,hdr.PixelDimensions,hdr.Offset);
end

