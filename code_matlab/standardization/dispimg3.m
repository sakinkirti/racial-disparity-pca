function fighandle = dispimg3(varargin)

if nargin>=3 && isa(varargin{3},'function_handle')
    feval(varargin{3},varargin{4:end});
    return;
else
    img = varargin{1};
end

fighandle = figure;   
dispimg(img(:,:,1),gcf);
slider = uicontrol('Style', 'slider',...
       'Max',size(img,3),'Min',1,...
       'Units', 'normalized', ...
       'Position', [.25 .005 .4 .04],...
       'SliderStep', [1/(size(img,3)-1) .5], ...
       'Value', 1, ...
       'Callback', {@dispimg3,@move_slider});
text_box = uicontrol('Style', 'text',...
       'Units', 'normalized', ...
       'Position', [.675 .006 .04 .04],...
       'String', '1');
udata.img = img;
udata.text_box = text_box;
set(gcf,'UserData',udata);

   
function move_slider
udata = get(gcf,'UserData');
img = udata.img;
val = round(get(gcbo,'Value'));
set(gcbo,'Value',val);
set(udata.text_box,'String',num2str(val));
save_xlim = xlim;
save_ylim = ylim;
dispimg(img(:,:,val),gcf);
xlim(save_xlim);
ylim(save_ylim);