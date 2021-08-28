clc
clear
cd 'C:/Users/USER/Desktop/SOF-VSR/TIP/data/test/'
video_list = dir('TVD');
%% BI
for scale = 2:4
    for idx_video = 3:length(video_list)
        mkdir(['TVD/',video_list(idx_video).name,'/lr_x', num2str(scale), '_BI']);            
        for idx_frame = 1:65
            if idx_frame<11
                img_hr = imread(['TVD/',video_list(idx_video).name,'/hr/hr',num2str(idx_frame-1, '%d'),'.png']);
   
            else
                img_hr = imread(['TVD/',video_list(idx_video).name,'/hr/hr',num2str(idx_frame-1, '%02d'),'.png']);
            end
            h = size(img_hr, 1);
            w = size(img_hr, 2);
            img_hr = img_hr(1:floor(h/scale/2)*scale*2, 1:floor(w/scale/2)*scale*2, :);                 
            img_lr = imresize(img_hr, 1/scale, 'bicubic');
            
            imwrite(img_lr, ['TVD/',video_list(idx_video).name,'/lr_x', num2str(scale), '_BI/lr_', num2str(idx_frame, '%02d'), '.png']);
        end
    end
end
