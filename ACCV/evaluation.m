clc
clear
%% evaluation on Vid4
addpath('metrics')
video_name = {'calendar','city','foliage','walk'};
psnr_vid4 = [];
ssim_vid4 = [];
for idx_video = 1:length(video_name)   
    psnr_video = [];
    ssim_video = [];
    for idx_frame = 3:29 % exclude the first and last 2 frames
        img_hr = imread(['data/',video_name{idx_video},'/hr/hr_', num2str(idx_frame,'%02d'),'.png']);
        img_sr = imread(['results/',video_name{idx_video},'/sr_', num2str(idx_frame,'%02d'),'.png']);
        
        img_hr_ycbcr = rgb2ycbcr(img_hr);
        img_hr_y = img_hr_ycbcr(:,:,1);
        img_sr_ycbcr = rgb2ycbcr(img_sr);
        img_sr_y = img_sr_ycbcr(:,:,1);
        
        psnr_video(idx_frame-2) = cal_psnr(img_sr_y, img_hr_y);
        ssim_video(idx_frame-2) = cal_ssim(img_sr_y, img_hr_y);
    end
    psnr_vid4(idx_video) = mean(psnr_video);
    ssim_vid4(idx_video) = mean(ssim_video);
    disp([video_name{idx_video},'---Mean PSNR: ', num2str(mean(psnr_video),'%0.4f'),', Mean SSIM: ', num2str(mean(ssim_video),'%0.4f')])
end
disp(['Vid4---Mean PSNR: ', num2str(mean(psnr_vid4),'%0.4f'),', Mean SSIM: ', num2str(mean(ssim_vid4),'%0.4f')])