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

from opt.train_opt import TrainOptions


if __name__ == '__main__':
    train_parser = TrainOptions()
    opt = train_parser.parse()

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
    transforms_train = transforms.Compose([transforms.Resize(image_size),
                                          transforms.ToTensor()])
    transforms_test = transforms.Compose([transforms.Resize(image_size),
                                          transforms.ToTensor()])

    trainset = HazyDataset(train_data_dir, transforms_train)
    testset = HazyDataset(test_data_dir, transforms_test)
    trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=num_threads)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_threads)

    print('Building models')
    if opt.model == 'residual_physics':
        net = ResidualPhysics('resnet')
    elif opt.model == 'two_stream':
        net = TwoStream()
    elif opt.model == 'dehazenet':
        pass
    elif opt.model == 'our_model':
        pass

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
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=beta)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # start training
    net.train()
    epoch_loss = []
    epoch_psnr = []
    epoch_ssim = []
    for epoch in range(epochs):
        batch_loss = []
        batch_psnr = []
        batch_ssim = []
        for i, data in enumerate(trainloader):
            rgb_input, nir_input = data['rgb'].to(device), data['nir'].to(device)
            rgb_dehazed, nir_dehazed = data['rgb_dehazed'].to(device), data['nir_dehazed'].to(device)
            rgb_gt = data['gt'].to(device)

            # zero gradient
            optimizer.zero_grad()

            # forward with physics solution
            outputs = net(rgb_input, nir_input, (rgb_dehazed, nir_dehazed))

            # calculate loss
            loss = criterion(outputs, rgb_gt)

            # backward
            loss.backward()

            # update weights
            optimizer.step()

            batch_loss.append(loss.item())
            batch_psnr.append(get_psnr_torch(rgb_gt, outputs))
            batch_ssim.append(get_ssim_torch(rgb_gt, outputs))

        current_epoch_loss = np.mean(batch_loss)
        epoch_loss.append(current_epoch_loss)
        lr_scheduler.step(current_epoch_loss)

        current_epoch_psnr = np.mean(batch_psnr)
        epoch_psnr.append(current_epoch_psnr)

        current_epoch_ssim = np.mean(batch_ssim)
        epoch_ssim.append(current_epoch_ssim)

        if current_epoch_loss < best_loss:
            best_loss = current_epoch_loss
            best_psnr = current_epoch_psnr
            best_ssim = current_epoch_ssim

        print('Epoch %d, training loss: %.5f, avg_psnr: %.2f, avg_ssim: %.2f' % (epoch+1, current_epoch_loss,
                                                                                 current_epoch_psnr, current_epoch_ssim))

        if (epoch+1) % opt.save_epoch_freq == 0:
            save_prefix = os.path.join(opt.save_dir, '%s'%opt.model)
            if not os.path.exists(save_prefix):
                os.mkdir(save_prefix)
            save_dir = os.path.join(save_prefix, '%02d_%s_%s.pth'%(opt.exp_id, opt.model, epoch+1))
            torch.save({'net': net.state_dict(),
                        'best_loss': best_loss,
                        'best_psnr': best_psnr,
                        'best_ssim': best_ssim}, save_dir)










