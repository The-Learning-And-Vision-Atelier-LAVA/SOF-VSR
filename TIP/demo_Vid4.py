from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_utils import TestsetLoader, ycbcr2rgb
from modules import SOFVSR
from torchvision.transforms import ToPILImage
import numpy as np
import os
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degradation", type=str, default='BI')
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--testset_dir', type=str, default='data/test/TVD')
    parser.add_argument('--chop_forward', type=bool, default=False)
    return parser.parse_args()


def chop_forward(x, model, scale, shave=16, min_size=5000, nGPUs=1):
    # divide into 4 patches
    b, n, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, :, 0:h_size, 0:w_size],
        x[:, :, :, 0:h_size, (w - w_size):w],
        x[:, :, :, (h - h_size):h, 0:w_size],
        x[:, :, :, (h - h_size):h, (w - w_size):w]]

    
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

    output = Variable(x.data.new(1, 1, h, w), volatile=True)
    output[:, :, 0:h_half, 0:w_half] = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


def main(cfg):
    # model
    net = SOFVSR(cfg, is_training=False)
    ckpt = torch.load('./log/' + cfg.degradation + '_x' + str(cfg.scale) + '.pth')
    net.load_state_dict(ckpt)
    torch.cuda.set_device(2)
    print(torch.cuda.current_device())
    if cfg.gpu_mode:
        net.cuda()
    
    with torch.no_grad():
        video_list = os.listdir(cfg.testset_dir)
        
        for idx_video in range(len(video_list)):
            video_name = video_list[idx_video]

            # dataloader
            test_set = TestsetLoader(cfg, video_name)
            test_loader = DataLoader(test_set, num_workers=1, batch_size=1, shuffle=False)

            for idx_iter, (LR_y_cube, SR_cb, SR_cr) in enumerate(test_loader):
                # data
                b, n_frames, h_lr, w_lr = LR_y_cube.size()
                LR_y_cube = Variable(LR_y_cube)
                LR_y_cube = LR_y_cube.view(b, -1, 1, h_lr, w_lr)

                if cfg.gpu_mode:
                    LR_y_cube = LR_y_cube.cuda()

                    if cfg.chop_forward:
                        # crop borders to ensure each patch can be divisible by 2
                        _, _, _, h, w = LR_y_cube.size()
                        h = int(h//16) * 16
                        w = int(w//16) * 16
                        LR_y_cube = LR_y_cube[:, :, :, :h, :w]
                        SR_cb = SR_cb[:, :h * cfg.scale, :w * cfg.scale]
                        SR_cr = SR_cr[:, :h * cfg.scale, :w * cfg.scale]
                        SR_y = chop_forward(LR_y_cube, net, cfg.scale).squeeze(0)

                    else:
                        SR_y = net(LR_y_cube).squeeze(0)

                else:
                    SR_y = net(LR_y_cube).squeeze(0)

                SR_y = np.array(SR_y.data.cpu())

                SR_ycbcr = np.concatenate((SR_y, SR_cb, SR_cr), axis=0).transpose(1,2,0)
                SR_rgb = ycbcr2rgb(SR_ycbcr) * 255.0
                SR_rgb = np.clip(SR_rgb, 0, 255)
                SR_rgb = ToPILImage()(np.round(SR_rgb).astype(np.uint8))

                # if not os.path.exists('results/Vid4'):
                #     os.mkdir('results/Vid4')
                # if not os.path.exists('results/Vid4/' + cfg.degradation + '_x' + str(cfg.scale)):
                #     os.mkdir('results/Vid4/' + cfg.degradation + '_x' + str(cfg.scale))
                # if not os.path.exists('results/Vid4/' + cfg.degradation + '_x' + str(cfg.scale) + '/' + video_name):
                #     os.mkdir('results/Vid4/' + cfg.degradation + '_x' + str(cfg.scale) + '/' + video_name)
                # SR_rgb.save('results/Vid4/' + cfg.degradation + '_x' + str(cfg.scale) + '/' + video_name + '/sr_' + str(idx_iter+2).rjust(2,'0') + '.png')

                #TVD
                print('come')
                if not os.path.exists('results/TVD'):
                    os.mkdir('results/TVD')
                if not os.path.exists('results/TVD/' + cfg.degradation + '_x' + str(cfg.scale)):
                    os.mkdir('results/TVD/' + cfg.degradation + '_x' + str(cfg.scale))
                if not os.path.exists('results/TVD/' + cfg.degradation + '_x' + str(cfg.scale) + '/' + video_name):
                    os.mkdir('results/TVD/' + cfg.degradation + '_x' + str(cfg.scale) + '/' + video_name)
                SR_rgb.save('results/TVD/' + cfg.degradation + '_x' + str(cfg.scale) + '/' + video_name + '/sr_' + str(idx_iter+2).rjust(2,'0') + '.png')


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
