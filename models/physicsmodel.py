from utils.darkchannel import *
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
from PIL import Image
from utils import get_psnr_torch, get_ssim_torch

def dcp_dehaze(I):
    window_size = 15
    t_threshold = 0.1
    window_size_guided_filter = 41
    epsilon = 1e-3

    if I.dtype == 'uint8':
        I = I/255.0

    I_erode, I_dark = get_dark_channel(I, window_size)
    L, _= get_atmosphere_light(I_dark, I)
    t_initial = init_transmission(I_erode, L)
    t = guided_filter(I, t_initial, window_size_guided_filter, epsilon)
    t_clip = clip_to_unit_range(t)
    J = inverse_model(I, t_clip, L, t_threshold)
    return clip_to_unit_range(J)


def rgb_nir_dcp(rgb_img, nir_img, patch_size=41):
    window_size = 15
    t_threshold = 0.1
    window_size_guided_filter = patch_size
    epsilon = 1e-3

    if rgb_img.dtype == 'uint8':
        rgb_img = rgb_img/255.0
    if nir_img.dtype == 'uint8':
        nir_img = nir_img/255.0

    rgb_erode, rgb_dark = get_dark_channel(rgb_img, window_size)
    A, _ = get_atmosphere_light(rgb_dark, rgb_img)
    t_rgb = init_transmission(rgb_erode, A)
    t_rgb = guided_filter(rgb_img, t_rgb, window_size_guided_filter, epsilon)
    t_rgb = clip_to_unit_range(t_rgb)
    t_nir = t_rgb**(1/8)
    J_rgb = inverse_model(rgb_img, t_rgb, A, t_threshold)
    J_nir = inverse_model(nir_img, t_nir, np.mean(A), t_threshold, False)

    return clip_to_unit_range(J_rgb), clip_to_unit_range(J_nir)


if __name__ == '__main__':
    import torch
    # do train first
    rgb_names = glob.glob('../dataset/test/RGB/*.tiff')
    psnr_all = []
    ssim_all = []
    for rgb_name in tqdm(rgb_names):
        elements = rgb_name.split('/')
        gt_name = os.path.join('../dataset/test', 'gt', '{}gt.tiff'.format(elements[-1][:-8]))
        rgb_image = Image.open(rgb_name)
        gt_image = np.asarray(Image.open(gt_name))/255.0
        rgb_dehazed = dcp_dehaze(np.asarray(rgb_image))
        psnr = get_psnr_torch(torch.from_numpy(rgb_dehazed), torch.from_numpy(gt_image))
        a = torch.from_numpy(rgb_dehazed).permute(2,0,1).unsqueeze(0)
        b = torch.from_numpy(gt_image).permute(2,0,1).unsqueeze(0)
        ssim = get_ssim_torch(a, b)
        psnr_all.append(psnr)
        ssim_all.append(ssim)
    print(np.mean(psnr_all), np.mean(ssim_all))

