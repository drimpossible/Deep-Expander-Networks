'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.expandergraphlayer import ExpanderConv2dReverse, ExpanderConv2d, channelShuffle

from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self,nClasses, width_mult, gtp, gsz, expsz):
        super(Net, self).__init__()
        self.nClasses = nClasses
        self.wm = width_mult

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride, gtype, groupsz=1, expandsz=1):
            if inp == 32 and gtype == 'full':
                inp = inp
                oup = int(oup*self.wm)
            elif oup == 1024 and inp==1024:
                inp = int(inp*self.wm)
                oup = oup
            elif inp == 128 and oup == 128 and gtype == 'expanderacc':
                gtype = 'expander'
            else:
                inp = int(inp*self.wm)
                oup = int(oup*self.wm)
            if gtype == 'full':
                layerdl = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
                layer = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
            elif gtype == 'groups':
                layerdl = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
                layer = nn.Conv2d(inp, oup, 1, 1, 0, groups=groupsz, bias=False)
            elif gtype == 'expander':
                layerdl = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
                layer = ExpanderConv2dReverse(inp, oup, kernel_size=1, stride=1, padding=0, expandSize=(oup//groupsz))
            elif gtype == 'shuffle':
                layerdl = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
                layer = nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, groups=groupsz, bias=False),channelShuffle(groupsz))
            elif gtype == 'expandershuffle':
                layerdl = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
                layer = nn.Sequential(ExpanderConv2dReverse(inp, oup, kernel_size=1, stride=1, padding=0, expandSize=(oup//groupsz)),channelShuffle(groupsz))
            elif gtype == 'expanderacc':
                layerdl = ExpanderConv2d(inp, inp, kernel_size=3, stride=stride, padding=1,  expandSize=expandsz)
                layer = ExpanderConv2dReverse(inp, oup, kernel_size=1, stride=1, padding=0, expandSize=(oup//groupsz))
            return nn.Sequential(
                layerdl,
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                layer,
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2),
            conv_dw( 32,  64, 1, 'full', gsz, expsz),
            conv_dw( 64, 128, 2, gtp, gsz, expsz),
            conv_dw(128, 128, 1, gtp, gsz, expsz),
            conv_dw(128, 256, 2, gtp, gsz, expsz),
            conv_dw(256, 256, 1, gtp, gsz, expsz),
            conv_dw(256, 512, 2, gtp, gsz, expsz),
            conv_dw(512, 512, 1, gtp, gsz, expsz),
            conv_dw(512, 512, 1, gtp, gsz, expsz),
            conv_dw(512, 512, 1, gtp, gsz, expsz),
            conv_dw(512, 512, 1, gtp, gsz, expsz),
            conv_dw(512, 512, 1, gtp, gsz, expsz),
            conv_dw(512, 1024, 2, gtp, gsz, expsz*3),
            conv_dw(1024, 1024, 1, gtp, gsz, expsz*3),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, nClasses)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        x = x.view(-1, self.nClasses)
        return x
