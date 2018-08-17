import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as modelzoo
import opts
import train
import utils
import models.__init__ as init_model
import mydatasets.__datainit__ as init_data
#from tensorboard_logger import Logger

parser = opts.myargparser()

def main():
    global opt, best_prec1

    opt = parser.parse_args()
    opt.logdir = opt.logdir+'/'+opt.name
    logger = None#Logger(opt.logdir)
    opt.lr = opt.maxlr

    print(opt)
    best_prec1 = 0
    cudnn.benchmark = True
    model = init_model.load_model(opt)
    if opt.model_def.startswith('alexnet') or opt.model_def.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    elif opt.ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    print(model)
    model, criterion, optimizer = init_model.setup(model,opt)

    trainer = train.Trainer(model, criterion, optimizer, opt, logger)
    validator = train.Validator(model, criterion, opt, logger)

    if opt.resume:
        if os.path.isfile(opt.resume):
            model, optimizer, opt, best_acc = init_model.resumer(opt, model, optimizer)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    dataloader = init_data.load_data(opt)
    train_loader = dataloader.train_loader
    #print(utils.get_mean_and_std(train_loader))
    val_loader = dataloader.val_loader

    for epoch in range(opt.start_epoch, opt.epochs):
        utils.adjust_learning_rate(opt, optimizer, epoch)
        print("Starting epoch number:",epoch,"Learning rate:", opt.lr)

        if opt.testOnly == False:
            trainer.train(train_loader, epoch, opt)
        if opt.tensorboard:
            logger.scalar_summary('learning_rate', opt.lr, epoch)

        prec1 = validator.validate(val_loader, epoch, opt)
        best_prec1 = max(prec1, best_prec1)
        init_model.save_checkpoint(opt, model, optimizer, best_prec1, epoch)

        print('Best Prec@1: [{0:.3f}]\t'.format(best_prec1))

if __name__ == '__main__':
    main()
