import torch
import torch.nn as nn
import numpy as np
import os
import time
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

from models.resphysics import ResidualPhysics
from models.twostream import TwoStream
from utils.dataloader import HazyDataset
from utils import get_psnr_torch, get_ssim_torch


def test(net, dataset, device, criterion, model):
    testloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loss = 0
    test_psnr = 0
    test_ssim = 0
    with torch.no_grad():
        net.eval()
        for inputs in testloader:
            if model == 'residual_physics':
                rgb_input, nir_input = inputs['rgb_input'].to(device), inputs['nir_input'].to(device)
                rgb_dehazed, nir_dehazed = inputs['rgb_dehazed'].to(device), inputs['nir_dehazed'].to(device)
                rgb_gt = inputs['gt'].to(device)
                outputs = net(rgb_input, nir_input, (rgb_dehazed, nir_dehazed))
            else:
                rgb_input, nir_input = inputs['rgb'].to(device), inputs['nir'].to(device)
                rgb_gt = inputs['gt'].to(device)
                outputs = net(rgb_input, nir_input)

            loss = criterion(rgb_gt, outputs)
            test_loss += loss.item()

            test_psnr += get_psnr_torch(rgb_gt, outputs)
            test_ssim += get_ssim_torch(rgb_gt, outputs)

        test_loss = test_loss/len(dataset)
        test_psnr = test_psnr/len(dataset)
        test_ssim = test_ssim/len(dataset)
    return test_loss, test_psnr, test_ssim