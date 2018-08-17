import torchvision
import re
import torch
import densenet_cifar as densenetcifar
from torch.autograd import Variable

global idx
idx = 0


def get_num_gen(gen):
    return sum(1 for x in gen)

def flops_layer(layer):
    """
    Calculate the number of flops for given a string information of layer.
    We extract only resonable numbers and use them.

    Args:
        layer (str) : example
            Linear (512 -> 1000)
            Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
    """


    idx_type_end = layer.find('(')
    type_name = layer[:idx_type_end]

    params = re.findall('[^a-z](\d+)', layer)
    paramsnum = 0

    if layer.find('Linear') >= 0:
        # global idx
        # H = alex[idx][0]
        # W = alex[idx][1]
        # idx += 1

        C1 = int(params[0])
        C2 = int(params[1])
        paramsnum = C1*C2

    elif layer.find('Conv2d') >= 0:
        global idx
        idx += 1

        C1 = int(params[0])
        C2 = int(params[1])
        K1 = int(params[2])
        K2 = int(params[3])
        paramsnum = C1*C2*K1*K2

    if type_name=="Conv2d":
        print(paramsnum)

    return paramsnum

def calculate_flops(gen):
    """
    Calculate the flops given a generator of pytorch model.
    It only compute the flops of forward pass.

    Example:
        >>> net = torchvision.models.resnet18()
        >>> calculate_flops(net.children())
    """
    paramsnum = 0;

    for child in gen:
        num_children = get_num_gen(child.children())

        if num_children == 0:
            paramsnum += flops_layer(str(child))

        else:
            paramsnum += calculate_flops(child.children())

    return paramsnum

net = densenetcifar.DenseNet3(58,100,60, 0.5,True,0)
# net = torchvision.models.alexnet()
paramsnum = calculate_flops(net.children())
print(paramsnum / 10**6, 'M')
