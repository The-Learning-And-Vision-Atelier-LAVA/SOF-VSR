clc
clear
cd("D:/")
%% evaluation on Vid4
addpath('metrics')
video_name = dir('cal/*');
disp(video_name)

scale = 4;
degradation = 'BI';
psnr_vid4 = [];
ssim_vid4 = [];
file = fopen('sof.txt', 'w');
for idx_video = 1:length(video_name)
    if strcmp(video_name(idx_video).name, '.') == 1 || strcmp(video_name(idx_video).name, '..') == 1
        continue
    end
    psnr_video = [];
    ssim_video = [];
    for idx_frame = 1:15 				% exclude the first and last 2 frames
        hrname = ['cal/' video_name(idx_video).name '/hr' num2str(idx_frame,'%d') '.png'];
        srname = ['cal/' video_name(idx_video).name '/sr_' num2str(idx_frame,'%d') '.png'];
        img_hr = imread(hrname);
        img_sr = imread(srname);
        
        h = min(size(img_hr, 1), size(img_sr, 1));
        w = min(size(img_hr, 2), size(img_sr, 2));
        
        border = 6 + scale;
        
        img_hr_ycbcr = rgb2ycbcr(img_hr);
        img_hr_y = img_hr_ycbcr(1+border:h-border, 1+border:w-border, 1);
        img_sr_ycbcr = rgb2ycbcr(img_sr);
        img_sr_y = img_sr_ycbcr(1+border:h-border, 1+border:w-border, 1);
        
        flag = 0;
        temp = cal_psnr(img_sr_y, img_hr_y);
        if isinf(temp) == 0 
            flag = 1;
            psnr_video(idx_frame) = temp;
            ssim_video(idx_frame) = cal_ssim(img_sr_y, img_hr_y);
        end
        
    end
    if flag == 1
        psnr_vid4(idx_video) = mean(psnr_video);
        ssim_vid4(idx_video) = mean(ssim_video);
        disp([video_name(idx_video).name,'---Mean PSNR: ', num2str(mean(psnr_video),'%0.4f'),', Mean SSIM: ', num2str(mean(ssim_video),'%0.4f')]);
        fprintf(file, [video_name(idx_video).name,'---Mean PSNR: ', num2str(mean(psnr_video),'%0.4f'),', Mean SSIM: ', num2str(mean(ssim_video),'%0.4f'), '\n\n']);
    end
    if flag == 0
        disp([video_name(idx_video).name, '---Mean PSNR: ', 'Cannot calculate because of Inf' ]);
        fprintf(file, [video_name(idx_video).name, '---Mean PSNR: ', 'Cannot calculate because of Inf', '\n\n' ]);
    end
        
end
disp(['---------------------------------------------'])
disp(['dataset ',degradation,'_x', num2str(scale) ,' SR---Mean PSNR: ', num2str(mean(psnr_vid4),'%0.4f'),', Mean SSIM: ', num2str(mean(ssim_vid4),'%0.4f')])
fprintf(file, ['dataset ',degradation,'_x', num2str(scale) ,' SR---Mean PSNR: ', num2str(mean(psnr_vid4),'%0.4f'),', Mean SSIM: ', num2str(mean(ssim_vid4),'%0.4f'), '\n\n']);
fclose(file);