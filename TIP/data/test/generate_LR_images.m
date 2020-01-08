clc
clear
video_list = dir('Vid4');
%% BI
for scale = 2:4
    for idx_video = 3:length(video_list)
        mkdir(['Vid4/',video_list(idx_video).name,'/lr_x', num2str(scale), '_BI']);            
        for idx_frame = 1:31
            img_hr = imread(['Vid4/',video_list(idx_video).name,'/hr/hr_',num2str(idx_frame, '%02d'),'.png']);
            h = size(img_hr, 1);
            w = size(img_hr, 2);
            img_hr = img_hr(1:floor(h/scale/2)*scale*2, 1:floor(w/scale/2)*scale*2, :);                 
            img_lr = imresize(img_hr, 1/scale, 'bicubic');
            
            imwrite(img_lr, ['Vid4/',video_list(idx_video).name,'/lr_x', num2str(scale), '_BI/lr_', num2str(idx_frame, '%02d'), '.png']);
        end
    end
end
%% BD
T = fspecial('gaussian', 13, 1.6);
for scale = 4
    for idx_video = 3:length(video_list)
        mkdir(['Vid4/',video_list(idx_video).name,'/lr_x', num2str(scale), '_BD']);
        for idx_frame = 1:31
            img_hr = imread(['Vid4/',video_list(idx_video).name,'/hr/hr_',num2str(idx_frame, '%02d'),'.png']);
            h = size(img_hr, 1);
            w = size(img_hr, 2);
            
            img_hr = img_hr(1:floor(h/scale/2)*scale*2, 1:floor(w/scale/2)*scale*2, :);
            img_hr = double(img_hr);
            img_lr = imfilter(img_hr, T, 'symmetric');
            img_lr = img_lr(1:scale:end, 1:scale:end, :);
            img_lr = uint8(img_lr);
            
            imwrite(img_lr,  ['Vid4/',video_list(idx_video).name,'/lr_x', num2str(scale), '_BD/lr_',  num2str(idx_frame, '%02d'), '.png']);
        end
    end
end
