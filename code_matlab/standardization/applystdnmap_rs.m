function outputvolume = applystdnmap_rs(inputvolume,standardization_map)
% OUTPUTVOLUME = applystdnmap(INPUTVOLUME, STANDARDIZATION_MAP)
% Applies STANDARDIZATION_MAP calculated via calcstdnmap to INPUTVOLUME,
% is the final step in intensity distribution standardization
%
% Satish Viswanath, May 2013
%
% SEE ALSO: int_stdn_landmarks.m, calcstdnmap.m

%Checks and balances
if ~ismatrix(standardization_map)
    error('STANDARDIZATION_MAP should be a N x 2 matrix');
end

%threshold the inputVolume
maxStdMap = max(standardization_map(:,1));
inputvolume = ceil(inputvolume);
inputvolume(inputvolume>maxStdMap) = maxStdMap;
inputvolints = unique(inputvolume);
outputvolume = inputvolume;
for i=2:length(inputvolints)
    %loc = find(ceil(standardization_map(:,1))==inputvolints(i));
    %outputvolume(inputvolume==inputvolints(i)) = standardization_map(loc,2); %#ok<FNDSB>
    outputvolume(inputvolume==inputvolints(i)) = standardization_map(inputvolints(i),2); 
end

end