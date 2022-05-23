% get the patient ids
destDir = '/Users/sakinkirti/Programming/Python/CCIPD/RacialDisparityDataset/to_nifti_cleaned';
cd(destDir)
patients = split(ls);

% iterate through the patients
for i=1:size(patients,1)
    % isolate the T2 and ADC images
    T2 = ['/', patients{i}, '/T2W.nii.gz'];
    ADC = ['/', patients{i}, '/ADC.nii.gz'];
    
    % run elastix on ADC and T2
    system(['/Users/sakinkirti/Programming/Python/CCIPD/code_matlab/elastix-5.0.1-mac/bin/elastix -f ' destDir ADC ' -m ' destDir T2 ' -p /Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/code_matlab/rigid.txt -out 'destDir filesep patients{i}]);
    
    % isolate the lesion masks and iterate through them
    cd([destDir,'/', patID]);
    lsArray = split(ls('LS*'));
    for j = 1:size(lsArray,1)
        
        lsMask = lsArray{j};
        system(['/Users/sakinkirti/Programming/Python/CCIPD/code_matlab/elastix-5.0.1-mac/bin/transformix -in ' destDir filesep patID filesep lsMask ' -tp ' destDir filesep patID '/TransformParameters.0.txt -out ' destDir filesep patID]);
        
        movefile([destDir filesep patID filesep 'result.nii.gz'], [destDir filesep patID filesep lsMask(1:end-7) 'ADC_mask.nii.gz']);
    end
end