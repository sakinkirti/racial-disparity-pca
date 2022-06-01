function img = contrast_stretch(img,lval,hval,scimg)

if nargin < 2
    lval = 0;
end
if nargin < 3
    hval = 255;
end
if nargin < 4
    scimg = img;
end

min_val = min(scimg(:));
max_val = max(scimg(:));

range = max_val-min_val;

if range == 0
    warning('scale factor of zero');
    return;
end

img = (img-min_val+lval)/range*hval;