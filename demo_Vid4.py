import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_utils import TestsetLoader, ycbcr2rgb
import numpy as np
from torchvision.transforms import ToPILImage
import os
import argparse
import torch.backends.cudnn as cudnn
from modules import SOFVSR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, default="calendar")
    parser.add_argument("--upscale_factor", type=int, default=4)
    parser.add_argument('--gpu_mode', type=bool, default=False)
    return parser.parse_args()

def main(cfg):
    video_name = cfg.video_name
    upscale_factor = cfg.upscale_factor
    use_gpu = cfg.gpu_mode

    test_set = TestsetLoader('data/'+ video_name, upscale_factor)
    test_loader = DataLoader(test_set, num_workers=1, batch_size=1, shuffle=False)
    net = SOFVSR(upscale_factor=upscale_factor)
    ckpt = torch.load('./log/SOFVSR_x' + str(upscale_factor) + '.pth')
    net.load_state_dict(ckpt)
    if use_gpu:
        net.cuda()


    for idx_iter, (LR_y_cube, SR_cb, SR_cr) in enumerate(test_loader):
        LR_y_cube = Variable(LR_y_cube)
        if use_gpu:
            LR_y_cube = LR_y_cube.cuda()
        SR_y = net(LR_y_cube)

        SR_y = np.array(SR_y.data)
        SR_y = SR_y[np.newaxis, :, :]

        SR_ycbcr = np.concatenate((SR_y, SR_cb, SR_cr), axis=0).transpose(1,2,0)
        SR_rgb = ycbcr2rgb(SR_ycbcr) * 255.0
        SR_rgb = np.clip(SR_rgb, 0, 255)
        SR_rgb = ToPILImage()(SR_rgb.astype(np.uint8))

        if not os.path.exists('results/' + video_name):
            os.mkdir('results/' + video_name)
        SR_rgb.save('results/'+video_name+'/sr_'+ str(idx_iter+2).rjust(2,'0') + '.png')

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)