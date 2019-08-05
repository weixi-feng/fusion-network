'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channels, block, num_blocks, symmetric=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layers = []
        for i, num_block in enumerate(num_blocks):
            self.layers.append(self._make_layer(block, 2**(i+6), num_block, stride=1))
        if symmetric:
            for i, num_block in enumerate(num_blocks[::-1]):
                self.layers.append(self._make_layer(block, 2**(len(num_blocks)+5-i), num_block, stride=1))
            self.layers.append(nn.Conv2d(64, input_channels, kernel_size=3, stride=1, padding=1, bias=False))
            self.layers.append(nn.ReLU())

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        return out


def ResNet18(input_channels, num_blocks=None, symmetric=True):
    if num_blocks is None:
        num_blocks = [2,2,2,2]
    return ResNet(input_channels, BasicBlock, num_blocks, symmetric)

def ResNet34(input_channels, num_blocks=None, symmetric=True):
    if num_blocks is None:
        num_blocks = [3,4,6,3]
    return ResNet(input_channels, BasicBlock, num_blocks, symmetric)

def ResNet50(input_channels, num_blocks=None, symmetric=True):
    if num_blocks is None:
        num_blocks = [3,4,6,3]
    return ResNet(input_channels, Bottleneck, num_blocks, symmetric)

def ResNet101(input_channels, num_blocks=None, symmetric=True):
    if num_blocks is None:
        num_blocks = [3,4,23,3]
    return ResNet(input_channels, Bottleneck, num_blocks, symmetric)

def ResNet152(input_channels, num_blocks=None, symmetric=True):
    if num_blocks is None:
        num_blocks = [3,8,36,3]
    return ResNet(input_channels, Bottleneck, num_blocks, symmetric)


if __name__ == '__main__':
    net = ResNet18(3, num_blocks=[2, 2, 2, 2])
    y = net(torch.randn(2, 3, 32, 32))
    print(y.size())