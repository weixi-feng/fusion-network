import torch
import torch.nn as nn
import numpy as np
import os
import time
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt

from models.resphysics import ResidualPhysics
from models.twostream import TwoStream
from models.onestream import OneStream
from models import get_model
from utils.dataloader import HazyDataset, Resize, ToTensor
from utils import get_psnr_torch, get_ssim_torch, prepare_data

from opt.train_opt import train_parser
from test import test


if __name__ == '__main__':
    opt = train_parser()

    # some hyper-parameters
    device = 'cuda' if opt.cuda and torch.cuda.is_available() else 'cpu'
    image_size = (opt.image_size, opt.image_size)
    lr = opt.lr
    batch_size = opt.batch_size
    epochs = opt.epoch
    niter = opt.niter
    niter_decay = opt.niter_decay
    beta = opt.beta1
    lr_policy = opt.lr_policy
    lr_decay_iters = opt.lr_decay_iters
    num_threads = opt.num_threads
    weight_decay = 0.9


    # recording parameter for training
    best_loss = float('inf')
    best_psnr = 0
    best_ssim = 0
    start_epoch = 0

    print('Preparing data')
    train_data_dir = opt.dataroot
    test_data_dir = os.path.join(*opt.dataroot.split('/')[:-1], 'test')
    trainset = prepare_data(opt.model, train_data_dir, image_size)
    testset = prepare_data(opt.model, test_data_dir, image_size)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    print('Building models')
    net = get_model(model=opt.model)
    net = net.to(device)

    if opt.load_model:
        print('loading model...')
        assert os.path.isdir(os.path.join(opt.save_dir, '%s'%opt.model))
        load_dir = os.path.join(opt.save_dir, '%s'%opt.model, '%02d_%s_%s.pth'%(opt.exp_id, opt.model, opt.load_epoch))
        checkpoint = torch.load(load_dir)
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['best_loss']
        best_psnr = checkpoint['best_psnr']
        best_ssim = checkpoint['best_ssim']
        start_epoch = opt.load_epoch

    criterion = nn.MSELoss()
    if opt.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    elif opt.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # start training
    print('Start training')
    epoch_loss = []
    epoch_psnr = []
    epoch_ssim = []
    for epoch in range(epochs):
        net.train()
        batch_loss = []
        batch_psnr = []
        batch_ssim = []
        for i, data in enumerate(trainloader):
            # zero gradient
            optimizer.zero_grad()

            if opt.model == 'residual_physics':
                rgb_input, nir_input = data['rgb_input'].to(device), data['nir_input'].to(device)
                rgb_dehazed, nir_dehazed = data['rgb_dehazed'].to(device), data['nir_dehazed'].to(device)
                rgb_gt = data['gt'].to(device)

                # forward with physics solution
                outputs = net(rgb_input, nir_input, (rgb_dehazed, nir_dehazed))
            else:
                rgb_input, nir_input = data['rgb_input'].to(device), data['nir_input'].to(device)
                rgb_gt = data['gt'].to(device)

                # forward with physics solution
                if opt.model == 'one_stream':
                    outputs = net(torch.cat((rgb_input, nir_input), dim=1))
                else:
                    outputs = net(rgb_input, nir_input)
            pdb.set_trace()
            # calculate loss
            loss = criterion(outputs, rgb_gt)

            # backward
            loss.backward()

            # update weights
            optimizer.step()

            batch_loss.append(loss.item())
            batch_psnr.append(get_psnr_torch(rgb_gt, outputs))
            batch_ssim.append(get_ssim_torch(rgb_gt, outputs))

            if epoch==20:
                plt.imshow(outputs[0].cpu().detach().numpy().transpose(1,2,0))
                plt.show()

        current_epoch_loss = np.mean(batch_loss)
        epoch_loss.append(current_epoch_loss)
        # lr_scheduler.step(current_epoch_loss)

        current_epoch_psnr = np.mean(batch_psnr)
        epoch_psnr.append(current_epoch_psnr)

        current_epoch_ssim = np.mean(batch_ssim)
        epoch_ssim.append(current_epoch_ssim)

        if current_epoch_loss < best_loss:
            best_loss = current_epoch_loss
            best_psnr = current_epoch_psnr
            best_ssim = current_epoch_ssim

        print('Epoch %d, training loss: %.5f, avg_psnr: %.2f, avg_ssim: %.4f' % (epoch+1, current_epoch_loss,
                                                                                current_epoch_psnr, current_epoch_ssim))
        if (epoch+1) % opt.test_freq == 0:
            test_loss, test_psnr, test_ssim = test(net, testset, device, criterion, opt.model)
            print('Testing results: avg_loss %.5f, avg_psnr: %.2f, avg_ssim %.4f' % (test_loss, test_psnr, test_ssim))

        if (epoch+1) % opt.save_epoch_freq == 0:
            save_prefix = os.path.join(opt.save_dir, '%s'%opt.model)
            if not os.path.exists(save_prefix):
                os.makedirs(save_prefix)
            save_dir = os.path.join(save_prefix, '%02d_%s_%s.pth'%(opt.exp_id, opt.model, epoch+1))
            torch.save({'net': net.state_dict(),
                        'best_loss': best_loss,
                        'best_psnr': best_psnr,
                        'best_ssim': best_ssim}, save_dir)










