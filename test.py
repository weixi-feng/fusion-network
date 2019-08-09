import torch
import torch.nn as nn
import numpy as np
import os
import time
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from PIL import Image

from models.resphysics import ResidualPhysics
from models.twostream import TwoStream
from utils.dataloader import HazyDataset, Resize
from utils import get_psnr_torch, get_ssim_torch
from opt.test_opt import test_parser
from models import get_model


def test(net, dataset, device, criterion, model):
    testloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loss = 0
    test_psnr = 0
    test_ssim = 0
    with torch.no_grad():
        net.eval()
        counter = 1
        for inputs in testloader:
            if model == 'residual_physics':
                rgb_input, nir_input = inputs['rgb_input'].to(device), inputs['nir_input'].to(device)
                rgb_dehazed, nir_dehazed = inputs['rgb_dehazed'].to(device), inputs['nir_dehazed'].to(device)
                rgb_gt = inputs['gt'].to(device)
                outputs = net(rgb_input, nir_input, (rgb_dehazed, nir_dehazed))
            else:
                rgb_input, nir_input = inputs['rgb_input'].to(device), inputs['nir_input'].to(device)
                rgb_gt = inputs['gt'].to(device)

                if model == 'one_stream':
                    outputs = net(torch.cat((rgb_input, nir_input), dim=1))
                else:
                    outputs = net(rgb_input, nir_input)

            if save_dir is not None:
                dehazed = outputs[0, ...].cpu().numpy()
                dehazed = Image.fromarray((dehazed * 255).astype('uint8'))
                dehazed.save(os.path.join(save_dir, '{0:05d}.tiff'.format(counter)))
                counter += 1

            loss = criterion(rgb_gt, outputs)
            test_loss += loss.item()

            test_psnr += get_psnr_torch(rgb_gt, outputs)
            test_ssim += get_ssim_torch(rgb_gt, outputs)

        test_loss = test_loss/len(dataset)
        test_psnr = test_psnr/len(dataset)
        test_ssim = test_ssim/len(dataset)
    return test_loss, test_psnr, test_ssim


if __name__ == '__main__':
    opt = test_parser()

    data_dir = opt.dataroot
    device = 'cuda' if opt.cuda and torch.cuda.is_available() else 'cpu'
    model = opt.model
    num_threads = opt.num_threads
    image_size = opt.image_size
    load_epoch = opt.load_epoch
    results_dir = opt.results_dir
    load_exp = opt.load_exp
    criterion = nn.MSELoss()

    model_prefix = os.path.join('./checkpoints', model)
    model_dir = os.path.join(model_prefix, '{0:02d}_{1}_{2}.pth'.format(load_exp, model, load_epoch))
    save_dir = os.path.join(results_dir, model)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Prepare data
    test_transforms = transforms.Compose([Resize((image_size, image_size)),
                                          transforms.ToTensor()])
    testset = HazyDataset(data_dir, test_transforms, dcp=False)

    net = get_model(model)
    net = net.to(device)

    checkpoints = torch.load(model_dir)
    net.load_state_dict(checkpoints['net'])

    test_loss, test_psnr, test_ssim = test(net, testset, device, criterion, model, save_dir)