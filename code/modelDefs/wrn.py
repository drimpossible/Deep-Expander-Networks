from __future__ import print_function
from __future__ import division

import math

# import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inChannels, outChannels, dropout=0.0, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inChannels)
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.downsample = downsample

    def forward(self, x):
        residual = F.relu(self.bn1(x))
        if self.downsample is not None:
            shortcut = self.downsample(residual)
        else:
            shortcut = x
        residual = self.conv1(residual)
        residual = self.drop(F.relu(self.bn2(residual)))
        residual = self.conv2(residual)
        out = residual + shortcut
        return out


class ResidualBlock(nn.Module):
    def __init__(self, block, inChannels, outChannels, depth, dropout=0.0, stride=1):
        super(ResidualBlock, self).__init__()
        if stride != 1 or inChannels != outChannels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inChannels, outChannels * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False)
            )
        else:
            downsample = None
        self.blocks = nn.Sequential()
        self.blocks.add_module('block0', block(inChannels, outChannels, dropout, stride, downsample))
        inChannels = outChannels * block.expansion
        for i in range(1, depth):
            self.blocks.add_module('block{}'.format(i), block(inChannels, outChannels, dropout))
            inChannels = outChannels * block.expansion

    def forward(self, x):
        return self.blocks(x)


class WRN(nn.Module):
    def __init__(self, depth, block='BasicBlock', widen=1, dropout=0.0, num_classes=10):
        super(WRN, self).__init__()
        assert (block in ['BasicBlock']), 'Unknown block type specified'
        if block == 'BasicBlock':
            assert (depth - 4) % 6 == 0, 'Depth should be 6n+4, and preferably one of 16, 22, 28, 40'
            n = (depth - 4) // 6
            block = BasicBlock
        self.widen = widen
        self.dropout = dropout
        self.num_classes = num_classes
        # self.initial = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True)
        # )
        self.initial = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.group1 = ResidualBlock(block, 16, 16 * self.widen, n, dropout=self.dropout)
        self.group2 = ResidualBlock(block, 16 * self.widen, 32 * self.widen, n, dropout=self.dropout, stride=2)
        self.group3 = ResidualBlock(block, 32 * self.widen, 64 * self.widen, n, dropout=self.dropout, stride=2)
        self.last = nn.BatchNorm2d(64 * self.widen)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * self.widen, self.num_classes)
        self.apply(self._weights_init)

    def forward(self, x):
        out = self.initial(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = F.relu(self.last(out))
        out = self.pool(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0.0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)
        elif classname.find('Linear') != -1:
            m.bias.data.fill_(0.0)
