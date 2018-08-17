import torch
import torch.nn as nn
import copy
import time
import shutil
import operator
import numpy as np
import random
import math

from PIL import Image, ImageOps
from torchvision import transforms

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def precision(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

"""
 Transform class to randomly rotate images
"""
class RandomRotate(object):
    def __call__(self, img):
        size = img.size
        angle = random.randint(-10, 10)
        img = img.rotate(angle, resample=Image.BICUBIC)
        img = img.resize(size, Image.ANTIALIAS)
        return img


class TenCrop(object):
    def __init__(self, size, opt):
        self.size = size
        self.opt = opt

    def __call__(self, img):
        centerCrop = transforms.CenterCrop(self.size)
        toPILImage = transforms.ToPILImage()
        toTensor = transforms.ToTensor()
        if self.opt.dataset == 'tuberlin':
            normalize = transforms.Normalize(mean=[0.06,], std=[0.93])
        if self.opt.dataset == 'sketchyrecognition':
            normalize = transforms.Normalize(mean=[0.0465,], std=[0.9])

        w, h = img.size(2), img.size(1)
        temp_output = []
        output = torch.FloatTensor(10, img.size(0), self.size, self.size)
        img = toPILImage(img)
        for img_cur in [img, img.transpose(Image.FLIP_LEFT_RIGHT)]:
            temp_output.append(centerCrop(img_cur))
            temp_output.append(img_cur.crop([0, 0, self.size, self.size]))
            temp_output.append(img_cur.crop([w-self.size, 0, w, self.size]))
            temp_output.append(img_cur.crop([0, h-self.size, self.size, h]))
            temp_output.append(img_cur.crop([w-self.size, h-self.size, w, h]))

        for img_idx in range(10):
            img_cur = temp_output[img_idx]
            img_cur = toTensor(img_cur)
            img_cur = normalize(img_cur)
            output[img_idx] = img_cur.view(img_cur.size(0), img_cur.size(1), img_cur.size(2))

        return output

def adjust_learning_rate(opt, optimizer, epoch):
    epoch = copy.deepcopy(epoch)
    lr = opt.maxlr
    if opt.learningratescheduler == 'decayschedular':
        while epoch >= opt.decayinterval:
            lr = lr*opt.decaylevel
            epoch = epoch - opt.decayinterval
    elif opt.learningratescheduler == 'imagenetschedular':
        lr = lr * (0.1 ** (epoch // 30))
    elif opt.learningratescheduler == 'cifarschedular':
        lr = lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))

    lr = max(lr,opt.minlr)
    opt.lr = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_mean_and_std(dataloader):
    '''Compute the mean and std value of dataset.'''
    mean = torch.zeros(3)
    std = torch.zeros(3)
    len_dataset = 0
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        len_dataset += 1
        for i in range(len(inputs[0])):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len_dataset)
    std.div_(len_dataset)
    return mean, std

def weights_init(model, opt):
    '''Add your favourite weight initializations.'''
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            #c  = math.sqrt(2.0 / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels))
            #m.weight.data = torch.randn(m.weight.data.size()).cuda() * c
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine == True:
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data = nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            #c =  math.sqrt(2.0 / m.weight.data.size(1));
            #m.weight.data = torch.randn(m.weight.data.size()).cuda() * c
            # TODO: Check bias
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
