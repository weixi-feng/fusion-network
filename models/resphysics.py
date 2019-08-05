import torch
import torch.nn as nn
from models.resnet import *
from models.unet import *

class Residual_Physics(nn.Module):
    def __init__(self, base_net='resnet'):
        super(Residual_Physics, self).__init__()
        if base_net == 'resnet':
            self.rgb_net = ResNet18(3, num_blocks=[2,2,2])
            self.nir_net = ResNet18(1, num_blocks=[2,2,2])
        elif base_net == 'unet':
            pass
        else:
            raise RuntimeError('No such option!')
        self.frontbone = nn.Sequential(nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU())

    def forward(self, rgb_image, nir_image):
        out1 = self.rgb_net(rgb_image)
        out2 = self.nir_net(nir_image)
        x = torch.cat((out1, out2), dim=1)
        out = self.frontbone(x)
        return out

