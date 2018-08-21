import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import models.alexnet as alexnet
import models.alexnetexpander1 as alexnetexpander1
import models.alexnetexpander2 as alexnetexpander2
import models.densenet_cifar as densenet_cifar
import models.densenetgrouped_cifar as densenetgrouped_cifar
import models.densenetexpander_cifar as densenetexpander_cifar
import models.densenet_imagenet as densenet
import models.densenetexpander_imagenet as densenetexpander
import models.vggcifar as vggcifar
import models.vggcifar_expander as vggcifarexpander
import models.resnet_imagenet as resnet
import models.resnetexpander_imagenet as resnetexpander
import models.mobilenet as mobilenet

import utils
import os
import shutil

def setup(model, opt):

    if opt.criterion == "l1":
        criterion = nn.L1Loss().cuda()
    elif opt.criterion == "mse":
        criterion = nn.MSELoss().cuda()
    elif opt.criterion == "crossentropy":
        criterion = nn.CrossEntropyLoss().cuda()
    elif opt.criterion == "hingeEmbedding":
        criterion = nn.HingeEmbeddingLoss().cuda()
    elif opt.criterion == "tripletmargin":
        criterion = nn.TripletMarginLoss(margin = opt.margin, swap = opt.anchorswap).cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if opt.optimType == 'sgd':
        optimizer = optim.SGD(parameters, lr = opt.lr, momentum = opt.momentum, nesterov = opt.nesterov, weight_decay = opt.weightDecay)
    elif opt.optimType == 'adam':
        optimizer = optim.Adam(parameters, lr = opt.maxlr, weight_decay = opt.weightDecay)

    if opt.weight_init:
        utils.weights_init(model, opt)

    return model, criterion, optimizer

def save_checkpoint(opt, model, optimizer, best_acc, epoch):

    state = {
        'epoch': epoch + 1,
        'arch': opt.model_def,
        'state_dict': model.state_dict(),
        'best_prec1': best_acc,
        'optimizer' : optimizer.state_dict(),
    }
    filename = "savedmodels/" + opt.model_def + '_' + opt.name + '_' + "best.pth.tar"

    torch.save(state, filename)

def resumer(opt, model, optimizer):

    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))

        return model, optimizer, opt, best_prec1


def load_model(opt):

    if opt.from_modelzoo:
        if opt.pretrained:
            print("=> using pre-trained model '{}'".format(opt.arch))
            model = models.__dict__[opt.model_def](pretrained=True)
        else:
            print("=> creating model '{}'".format(opt.arch))
            model = models.__dict__[opt.model_def]()

        return model
    else:
        if opt.pretrained_file != '':
            model = torch.load(opt.pretrained_filedir)
        else:
            if opt.model_def == 'alexnet':
                model = alexnet.Net()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'mobilenet':
                model = mobilenet.Net(nClasses=opt.nclasses, width_mult=opt.widthmult, gtp=opt.grouptype, gsz=opt.sp, expsz=opt.exp)
                if opt.cuda:
                    model = model.cuda()

            elif opt.model_def == 'alexnetexpander':
                model = alexnetexpander.Net()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'vgg16cifar':
                model = vggcifar.vgg16()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'vgg16cifar_bn':
                model = vggcifar.vgg16_bn()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'vgg16cifarexpander':
                model = vggcifarexpander.vgg16()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'vgg16cifar_bnexpander':
                model = vggcifarexpander.vgg16_bn()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'densenet_cifar':
                model = densenet_cifar.DenseNet3(opt.layers, opt.nclasses, opt.growth, reduction=opt.reduce,
                bottleneck=opt.bottleneck, dropRate=opt.droprate)
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'densenetgrouped_cifar':
                model = densenetgrouped_cifar.DenseNet3(opt.layers, opt.nclasses, opt.growth, reduction=opt.reduce,
                bottleneck=opt.bottleneck, dropRate=opt.droprate)
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'densenetexpander_cifar':
                model = densenetexpander_cifar.DenseNet3(opt.layers, opt.nclasses, opt.growth, reduction=opt.reduce,
                bottleneck=opt.bottleneck, dropRate=opt.droprate, expandSize=opt.expandSize)
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'densenet121':
                model = densenet.densenet121()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'densenet169':
                model = densenet.densenet169()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'densenet161':
                model = densenet.densenet161()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'densenet201':
                model = densenet.densenet201()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'densenetexpander121':
                model = densenetexpander.densenet121(expandSize=opt.expandSize)
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'densenetexpander169':
                model = densenetexpander.densenet169(expandSize=opt.expandSize)
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'densenetexpander161':
                model = densenetexpander.densenet161(expandSize=opt.expandSize)
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'densenetexpander201':
                model = densenetexpander.densenet201(expandSize=opt.expandSize)
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'resnet34':
                model = resnet.resnet34()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'resnet50':
                model = resnet.resnet50()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'resnet101':
                model = resnet.resnet101()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'resnet152':
                model = resnet.resnet152()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'resnetexpander34':
                model = resnetexpander.resnet34(opt.expandSize)
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'resnetexpander50':
                model = resnetexpander.resnet50(opt.expandSize)
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'resnetexpander101':
                model = resnetexpander.resnet101(opt.expandSize)
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'resnetexpander152':
                model = resnetexpander.resnet152(opt.expandSize)
                if opt.cuda:
                    model = model.cuda()

            elif opt.model_def == 'resnet18':
                model = resnet.resnet18()
                if opt.cuda:
                    model = model.cuda()
            elif opt.model_def == 'resnet50':
                model = resnet.resnet50()
                if opt.cuda:
                    model = model.cuda()

    return model
