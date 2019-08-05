import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import glob


class HazyNIRDataset(Dataset):
    def __init__(self, rgb_path, nir_path, gt_path, image_size):
        self.rgb_path = rgb_path
        self.nir_path = nir_path
        self.gt_path = gt_path
        self.size = image_size
        self.rgb_names = glob.glob(os.path.join(self.rgb_path, '*.tiff'))

    def __getitem__(self, idx):
        rgb_image_name = self.rgb_names[idx]
        components = rgb_image_name.split('/')
        nir_image_name = os.path.join(self.nir_path, '%snir.tiff' % components[-1][:-8])
        gt_image_name = os.path.join(self.gt_path, '%sgt.tiff' % components[-1][:-8])
        rgb_image = Image.open(rgb_image_name)
        nir_image = Image.open(nir_image_name)
        gt_image = Image.open(gt_image_name)

        rgb_image = TF.to_tensor(TF.resize(rgb_image, self.size))
        nir_image = TF.to_tensor(TF.resize(nir_image, self.size))
        gt_image = TF.to_tensor(TF.resize(gt_image, self.size))

        image_dict = {'rgb': rgb_image,
                      'nir': nir_image,
                      'gt': gt_image}
        return image_dict

    def __len__(self):
        return len(self.image_names)