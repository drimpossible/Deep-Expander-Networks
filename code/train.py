import torch
from torch.autograd import Variable
from utils import AverageMeter
from utils import precision
import torch.nn as nn

import utils
import torch.nn.functional as F
import math
import time

class Trainer():
    def __init__(self, model, criterion, optimizer, opt, logger):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

    def train(self, trainloader, epoch, opt):

        self.model.train()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()
        self.data_time.reset()
        self.batch_time.reset()

        end = time.time()
        for i, (input, target) in enumerate(trainloader, 0):
            if opt.cuda:
                input = input.cuda(async=True)
                target = target.cuda(async=True)

            input_var = Variable(input)
            target_var = Variable(target)

            self.data_time.update(time.time() - end)
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            #Broadcast based function. Adapted from:- https://github.com/pytorch/examples/blob/master/imagenet/main.py
            prec1, prec5 = precision(output.data, target, topk=(1,5))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.losses.update(loss.data[0], input.size(0))
            self.top1.update(prec1[0], input.size(0))
            self.top5.update(prec5[0], input.size(0))

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            if opt.verbose == True and i % opt.printfreq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f} ({batch_time.sum:.3f})\t'
                      'Data {data_time.avg:.3f} ({data_time.sum:.3f})\t'
                      'Loss {loss.avg:.3f}\t'
                      'Prec@1 {top1.avg:.4f}\t'
                      'Prec@5 {top5.avg:.4f}'.format(
                       epoch, i, len(trainloader), batch_time=self.batch_time,
                       data_time=self.data_time, loss=self.losses,
                       top1=self.top1, top5=self.top5))

            # log to TensorBoard
        if opt.tensorboard:
            self.logger.scalar_summary('train_loss', self.losses.avg, epoch)
            self.logger.scalar_summary('train_acc', self.top1.avg, epoch)

        print('Train: [{0}]\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss {loss.avg:.3f}\t'
              'Prec@1 {top1.avg:.4f}\t'
              'Prec@5 {top5.avg:.4f}\t'.format(
               epoch, batch_time=self.batch_time,
               data_time= self.data_time, loss=self.losses,
               top1=self.top1, top5=self.top5))


class Validator():
    def __init__(self, model, criterion, opt, logger):

        self.model = model
        self.criterion = criterion
        self.logger = logger
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

    def validate(self, valloader, epoch, opt):
        self.model.eval()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()
        self.data_time.reset()
        self.batch_time.reset()
        end = time.time()

        for i, (input, target) in enumerate(valloader, 0):
            if opt.tenCrop:
                input = input.view(input.size(0)*input.size(1), input.size(2), input.size(3), input.size(4))

            if opt.cuda:
                input = input.cuda(async=True)
                target = target.cuda(async=True)

            input_var = Variable(input, volatile=True)
            target_var = Variable(target, volatile=True)

            self.data_time.update(time.time() - end)
            output = self.model(input_var)

            if opt.tenCrop:
                # print("Doing 10crop")
                output = output.view(output.size(0) // 10, 10, output.size(1)).sum(1).squeeze(1).div(10.0)

            loss = self.criterion(output, target_var)

            prec1, prec5 = precision(output.data, target, topk=(1,5))
            self.losses.update(loss.data[0], input.size(0))

            self.top1.update(prec1[0], input.size(0))
            self.top5.update(prec5[0], input.size(0))

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

        # log to TensorBoard
        if opt.tensorboard:
            self.logger.scalar_summary('val_loss', self.losses.avg, epoch)
            self.logger.scalar_summary('val_acc', self.top1.avg, epoch)

        print('Val: [{0}]\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss {loss.avg:.3f}\t'
              'Prec@1 {top1.avg:.4f}\t'
              'Prec@5 {top5.avg:.4f}\t'.format(
               epoch, batch_time=self.batch_time,
               data_time= self.data_time, loss=self.losses,
               top1=self.top1, top5=self.top5))

        return self.top1.avg
