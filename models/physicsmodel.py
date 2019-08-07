from utils.darkchannel import *
import numpy as np
import matplotlib.pyplot as plt


def dcp_dehaze(I):
    window_size = 15
    t_threshold = 0.1
    window_size_guided_filter = 41
    epsilon = 1e-3

    I_erode, I_dark = get_dark_channel(I, window_size)
    L, _= get_atmosphere_light(I_dark, I)
    t_initial = init_transmission(I_erode, L)
    t = guided_filter(I, t_initial, window_size_guided_filter, epsilon)
    t_clip = clip_to_unit_range(t)
    J = inverse_model(I, t_clip, L, t_threshold)
    return clip_to_unit_range(J)


def rgb_nir_dcp(rgb_img, nir_img):
    window_size = 15
    t_threshold = 0.1
    window_size_guided_filter = 41
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
    rgb_img = plt.imread('../dataset/RGB/01_00001_rgb.tiff')
    nir_img = plt.imread('../dataset/NIR/01_00001_nir.tiff')
    rgb_img = rgb_img/255.0
    nir_img = nir_img/255.0
    J_rgb, J_nir = rgb_nir_dcp(rgb_img, nir_img)
    plt.imshow(J_rgb, cmap='gray')
    plt.show()