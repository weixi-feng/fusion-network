import torch
import torch.nn as nn
from models.resnet import ResNet18
from models.unet import *


class ResidualPhysics(nn.Module):
    def __init__(self, base_net='resnet'):
        super(ResidualPhysics, self).__init__()
        if base_net == 'resnet':
            self.rgb_net = ResNet18(3, num_blocks=[2,2,2], symmetric=True)
            self.nir_net = ResNet18(1, num_blocks=[2,2,2], symmetric=True)
            # self.rgb_physics = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1)
            # self.nir_physics = nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1)
        elif base_net == 'unet':
            pass
        else:
            raise RuntimeError('No such option!')
        self.frontbone = nn.Sequential(nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU())

    def forward(self, rgb_image, nir_image, phy_sol):
        up_stream = self.rgb_net(rgb_image)
        down_stream = self.nir_net(nir_image)

        rgb_sol, nir_sol = phy_sol
        up_stream += rgb_sol
        down_stream += nir_sol

        x = torch.cat((up_stream, down_stream), dim=1)
        out = self.frontbone(x)
        return out

if __name__ == '__main__':
    net = ResidualPhysics()
    a = torch.randn(1, 3, 480, 640)
    b = torch.randn(1, 1, 480, 640)
    out = net(a, b, (a,b))
    print(out.size())