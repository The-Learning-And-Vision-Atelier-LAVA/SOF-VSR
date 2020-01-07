import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_utils import TestsetLoader, ycbcr2rgb
import numpy as np
from torchvision.transforms import ToPILImage
import os
import argparse
from modules import SOFVSR
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, default="calendar")
    parser.add_argument("--upscale_factor", type=int, default=4)
    parser.add_argument('--gpu_mode', type=bool, default=False)
    parser.add_argument('--chop_forward', type=bool, default=False)
    return parser.parse_args()

def chop_forward(x, model, scale, shave=16, min_size=5000, nGPUs=1):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
            output_batch = model(input_batch)
            outputlist.append(output_batch.data)
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = Variable(x.data.new(h, w), volatile=True)
    output[0:h_half, 0:w_half] = outputlist[0][0:h_half, 0:w_half]
    output[0:h_half, w_half:w] = outputlist[1][0:h_half, (w_size - w + w_half):w_size]
    output[h_half:h, 0:w_half] = outputlist[2][(h_size - h + h_half):h_size, 0:w_half]
    output[h_half:h, w_half:w] = outputlist[3][(h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output
	
def main(cfg):
    video_name = cfg.video_name
    upscale_factor = cfg.upscale_factor
    use_gpu = cfg.gpu_mode

    test_set = TestsetLoader('data/test/'+ video_name, upscale_factor)
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
            if cfg.chop_forward:
                # crop borders to ensure each patch can be divisible by 2
                _, _, h, w = LR_y_cube.size()
                h = int(h//16) * 16
                w = int(w//16) * 16
                LR_y_cube = LR_y_cube[:, :, :h, :w]
                SR_cb = SR_cb[:, :h * upscale_factor, :w * upscale_factor]
                SR_cr = SR_cr[:, :h * upscale_factor, :w * upscale_factor]
                SR_y = chop_forward(LR_y_cube, net, cfg.upscale_factor)
            else:
                SR_y = net(LR_y_cube)
	    SR_y = SR_y.cpu()
        else:
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
