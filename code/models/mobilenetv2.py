import torch.nn as nn
import math
from models.layers.expandergraphlayer import ExpanderConv2dReverse, ExpanderConv2d, channelShuffle


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, gtype, groupsz1=1, groupsz2=1, expandsz=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        if gtype=="full":
            layer1 = nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False)
            layer2 = nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False)
            layerdl = nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False)
        elif gtype=="groups":
            layer1 = nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, groups=groupsz1, bias=False)
            layer2 = nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, groups=groupsz2, bias=False)
            layerdl = nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False)
        elif gtype=="shuffle":
            layer1 = nn.Sequential(nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, groups=groupsz1, bias=False),channelShuffle(groupsz1))
            layer2 = nn.Sequential(nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, groups=groupsz2, bias=False),channelShuffle(groupsz2))
            layerdl = nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False)
        elif gtype=="expander":
            layer1 = ExpanderConv2dReverse(inp, inp * expand_ratio, kernel_size=1, stride=1, padding=0, expandSize=((inp * expand_ratio)//groupsz1))
            layer2 = ExpanderConv2d(inp * expand_ratio, oup, kernel_size=1, stride=1, padding=0, expandSize=((inp * expand_ratio)//groupsz2))
            layerdl = nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False)
        elif gtype=="expanderacc":
            layer1 = ExpanderConv2dReverse(inp, inp * expand_ratio, kernel_size=1, stride=1, padding=0, expandSize=((inp * expand_ratio)//groupsz1))
            layer2 = ExpanderConv2d(inp * expand_ratio, oup, kernel_size=1, stride=1, padding=0, expandSize=((inp * expand_ratio)//groupsz2))
            layerdl = ExpanderConv2d(inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=stride, padding=1, expandSize=expandsz)
        self.conv = nn.Sequential(
            # pw
            layer1,
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            layerdl,
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            layer2,
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Net(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., gtype='expanderacc', hsp=1, lsp=1, lexp=1, hexp=1):
        super(Net, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1, "full", 1, 1 ,1],
            [6, 24, 2, 2, gtype, hsp, hsp, 1],
            [6, 32, 1, 2, gtype, hsp, lsp, 1],
            [6, 32, 2, 1, gtype, lsp, lsp, 1],
            [6, 64, 1, 2, gtype, lsp, 1, hexp],
            [6, 64, 3, 1, gtype, lsp, lsp, lexp],
            [6, 96, 1, 1, gtype, lsp, hsp, lexp],
            [6, 96, 2, 1, gtype, hsp, hsp, lexp],
            [6, 160, 3, 2, gtype, hsp, hsp, hexp],
            [6, 320, 1, 1, gtype, hsp, hsp, hexp],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s, typ, gs1, gs2, expsz in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t, typ, gs1, gs2, expsz))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t, typ, gs1, gs2, expsz))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(input_size//32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
