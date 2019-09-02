import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from modules import SOFVSR, optical_flow_warp
import argparse
from data_utils import TrainsetLoader
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upscale_factor", type=int, default=4)
    parser.add_argument('--gpu_mode', type=bool, default=False)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_iters', type=int, default=300000, help='number of iterations to train')
    parser.add_argument('--trainset_dir', type=str, default='data/train')
    return parser.parse_args()

def main(cfg):
    use_gpu = cfg.gpu_mode
    net = SOFVSR(cfg.upscale_factor, is_training=True)
    if use_gpu:
        net.cuda()
    cudnn.benchmark = True

    train_set = TrainsetLoader(cfg.trainset_dir, cfg.upscale_factor, cfg.patch_size, cfg.n_iters*cfg.batch_size)
    train_loader = DataLoader(train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)

    # train
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    criterion_L2 = torch.nn.MSELoss()
    if use_gpu:
        criterion_L2 = criterion_L2.cuda()
    milestones = [50000, 100000, 150000, 200000, 250000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    loss_list = []

    for idx_iter, (LR, HR) in enumerate(train_loader):
        scheduler.step()

        LR, HR = Variable(LR), Variable(HR)
        if use_gpu:
            LR = LR.cuda()
            HR = HR.cuda()

        (res_01_L1, res_01_L2, flow_01_L1, flow_01_L2, flow_01_L3), (
            res_21_L1, res_21_L2, flow_21_L1, flow_21_L2, flow_21_L3), SR = net(LR)
        warped_01 = optical_flow_warp(torch.unsqueeze(HR[:, 0, :, :], dim=1), flow_01_L3)
        warped_21 = optical_flow_warp(torch.unsqueeze(HR[:, 2, :, :], dim=1), flow_21_L3)

        # losses
        loss_SR = criterion_L2(SR, torch.unsqueeze(HR[:, 1, :, :], 1))
        loss_OFR_1 = 1 * (criterion_L2(warped_01, torch.unsqueeze(HR[:, 1, :, :], 1)) + 0.01 * L1_regularization(flow_01_L3)) + \
                     0.25 * (torch.mean(res_01_L2 ** 2) + 0.01 * L1_regularization(flow_01_L2)) + \
                     0.125 * (torch.mean(res_01_L1 ** 2) + 0.01 * L1_regularization(flow_01_L1))
        loss_OFR_2 = 1 * (criterion_L2(warped_21, torch.unsqueeze(HR[:, 1, :, :], 1)) + 0.01 * L1_regularization(flow_21_L3)) + \
                     0.25 * (torch.mean(res_21_L2 ** 2) + 0.01 * L1_regularization(flow_21_L2)) + \
                     0.125 * (torch.mean(res_21_L1 ** 2) + 0.01 * L1_regularization(flow_21_L1))
        loss = loss_SR + 0.01 * (loss_OFR_1 + loss_OFR_2) / 2
        loss_list.append(loss.data.cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save checkpoint
        if idx_iter % 5000 == 0:
            print('Iteration---%6d,   loss---%f' % (idx_iter + 1, np.array(loss_list).mean()))
            torch.save(net.state_dict(), 'log/BI_x' + str(cfg.upscale_factor) + '_iter' + str(idx_iter) + '.pth')
            loss_list = []

def L1_regularization(image):
    b, _, h, w = image.size()
    reg_x_1 = image[:, :, 0:h-1, 0:w-1] - image[:, :, 1:, 0:w-1]
    reg_y_1 = image[:, :, 0:h-1, 0:w-1] - image[:, :, 0:h-1, 1:]
    reg_L1 = torch.abs(reg_x_1) + torch.abs(reg_y_1)
    return torch.sum(reg_L1) / (b*(h-1)*(w-1))

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)







