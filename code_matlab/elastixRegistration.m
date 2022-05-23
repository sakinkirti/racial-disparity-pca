% get the patient ids
destDir = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/to_nifti_cleaned';
cd(destDir)
patients = split(ls);

% store the elastix, transformix, and rigid pathnames
elastix = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/elastix-5.0.0-mac/bin/elastix';
transformix = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/elastix-5.0.0-mac/bin/transformix';
rigid = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/rigid.txt';

% iterate through the patients
for i=1:size(patients,1)-1
    % isolate the T2 and ADC images
    T2W = ['/', patients{i}, '/T2W.nii.gz'];
    ADC = ['/', patients{i}, '/ADC.nii.gz'];
    
    % run elastix on ADC and T2
    system([elastix ' -f ' destDir ADC ' -m ' destDir T2W ' -p ' rigid ' -out ' destDir filesep patients{i}])
    
    % isolate the lesion masks and iterate through them
    cd([destDir,'/', patients{i}]);
    lsArray = split(ls('LS*'));
    for j = 1:size(lsArray,1)-1
        
        lsMask = lsArray{j};
        system([transformix ' -in ' destDir filesep patients{i} filesep lsMask ' -tp ' destDir filesep patients{i} '/TransformParameters.0.txt -out ' destDir filesep patients{i}]);
        
        % rename the file
        movefile([destDir filesep patients{i} filesep 'result.mha'], [destDir filesep patients{i} filesep lsMask(1:end-7) '_ADC_mask.mha']);
    end
end

% move back to the main dir at the end of the program
cd('/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab')