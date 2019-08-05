import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import glob
from models.physicsmodel import rgb_nir_dcp


class HazyDataset(Dataset):
    def __init__(self, prefix, transform=None):
        self.rgb_path = os.path.join(prefix, 'RGB')
        self.nir_path = os.path.join(prefix, 'NIR')
        self.gt_path = os.path.join(prefix, 'gt')
        self.transform = transform
        self.rgb_names = glob.glob(os.path.join(self.rgb_path, '*.tiff'))

    def __getitem__(self, idx):
        rgb_image_name = self.rgb_names[idx]
        components = rgb_image_name.split('/')
        nir_image_name = os.path.join(self.nir_path, '%snir.tiff' % components[-1][:-8])
        gt_image_name = os.path.join(self.gt_path, '%sgt.tiff' % components[-1][:-8])

        rgb_image = Image.open(rgb_image_name)
        nir_image = Image.open(nir_image_name)
        gt_image = Image.open(gt_image_name)

        rgb_dehazed, nir_dehazed = rgb_nir_dcp(np.asarray(rgb_image), np.asarray(nir_image))

        if self.transform is not None:
            rgb_image = self.transform(rgb_image)
            nir_image = self.transform(nir_image)
            rgb_dehazed = self.transform(rgb_dehazed)
            nir_dehazed = self.transform(nir_dehazed)
            gt_image = self.transform(gt_image)

        image_dict = {'rgb': rgb_image,
                      'nir': nir_image,
                      'rgb_dehazed': rgb_dehazed,
                      'nir_dehazed': nir_dehazed,
                      'gt': gt_image}

        return image_dict

    def __len__(self):
        return len(self.rgb_names)
