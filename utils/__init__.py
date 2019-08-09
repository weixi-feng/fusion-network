import torch
import torch.nn.functional as F
from math import log10
from skimage.measure import compare_ssim
from utils.pytorch_ssim import ssim
from utils.dataloader import HazyDataset


def get_psnr_torch(tensor_1, tensor_2):
    mse = F.mse_loss(tensor_1, tensor_2)
    psnr = 10 * log10(1 / mse.item())
    return psnr


def get_ssim_torch(tensor1, tensor2):
    torch_ssim = ssim(tensor1, tensor2)
    return torch_ssim.item()


def prepare_data(model, data_dir, image_size, transform):
    if model == 'residual_physics':
        dataset = HazyDataset(data_dir, image_size, transform, True)
    else:
        dataset = HazyDataset(data_dir, image_size, transform, False)
    return dataset