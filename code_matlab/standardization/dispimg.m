function f = dispimg(varargin)

if nargin>=3 && isa(varargin{3},'function_handle')
    feval(varargin{3},varargin{4:end});
    return;
else
    img = varargin{1};
    if nargin >= 3
        x_lim = varargin{2};
        y_lim = varargin{3};
        xwdth = diff(x_lim)/size(img,2);
        ywdth = diff(y_lim)/size(img,1);
    else
        x_lim = [1 size(img,2)];
        y_lim = [1 size(img,1)];
        xwdth = 1;
        ywdth = 1;
    end
    if nargin == 2 || nargin == 4
        f = varargin{end};
    else
        f = figure;
    end
end

udata.img = img;
img = contrast_stretch(img,0,255);

cmap = repmat(linspace(0,1,256)',1,3);
colormap(cmap);
udata.img_handle = image(x_lim,y_lim,img);
axis equal;
axis tight;

udata.x_lim = x_lim;
udata.y_lim = y_lim;
udata.xwdth = xwdth;
udata.ywdth = ywdth;
set(gca,'UserData',udata);
set(gcf,'WindowButtonDownFcn',{@dispimg,@button_callback});

toggle = uicontrol('Style', 'togglebutton',...
       'Units', 'normalized', ...
       'Position', [0 0 .05 .05],...
       'String', 'Txt', ...
       'Callback', {@dispimg,@toggle_callback});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function button_callback
pts1 = get(gca,'CurrentPoint');  
rbbox; 
pts2 = get(gca,'CurrentPoint');           
min_pts = min(pts1(1,1:2),pts2(1,1:2));
max_pts = max(pts1(1,1:2),pts2(1,1:2));
udata = get(gca,'UserData');
hold = prod(max_pts-min_pts) > 9*udata.xwdth*udata.ywdth;

if hold && strcmp(get(gcf,'SelectionType'),'normal')
    xlim([min_pts(1) max_pts(1)]);
    ylim([min_pts(2) max_pts(2)]);
elseif ~hold && strcmp(get(gcf,'SelectionType'),'alt')
    subimg_size = diff([xlim' ylim']); 
    xlim(max(min(xlim + [-1 1]*.25*subimg_size(1),udata.x_lim(2)),udata.x_lim(1)));
    ylim(max(min(ylim + [-1 1]*.25*subimg_size(2),udata.y_lim(2)),udata.y_lim(1)));
elseif hold && strcmp(get(gcf,'SelectionType'),'alt')
    img = get(udata.img_handle,'CData');
    sxl = round( ([min_pts(1) max_pts(1)] - udata.x_lim(1))/udata.xwdth ) + 1;
    syl = round( ([min_pts(2) max_pts(2)] - udata.y_lim(1))/udata.ywdth ) + 1;
    simg = img(syl(1):syl(2),sxl(1):sxl(2));
    img = contrast_stretch(img,0,255,simg);
    set(get(gca,'Children'),'CData',img);
elseif strcmp(get(gcf,'SelectionType'),'open')
    xlim(udata.x_lim)
    ylim(udata.y_lim);
end

%%%%%%%%%%%%%%%%%%%%%%%
function toggle_callback
udata = get(gca,'Userdata');
if get(gcbo,'Value') == 1
    sxl = round( (xlim - udata.x_lim(1))/udata.xwdth + [.75 -.75] ) + 1;
    syl = round( (ylim - udata.y_lim(1))/udata.ywdth + [.6 -.6] ) + 1;
    txt_mtx = udata.img(syl(1):syl(2),sxl(1):sxl(2));
    [x,y] = meshgrid(sxl(1):sxl(2),syl(1):syl(2));
    if numel(x) > 400
        set(gcbo,'Value',0);
        return;
    end
    udata.txt_handle = text(x(:),y(:),num2str(txt_mtx(:),5),...
        'HorizontalAlignment','center','Color','blue','FontSize',8);
    set(gca,'UserData',udata);
else
    delete(udata.txt_handle);
end
    
    
    
    