import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os

class CIFAR10():
    def __init__(self, opt):
        kwargs = {
          'num_workers': opt.workers,
          'batch_size' : opt.batchSize,
          'shuffle' : True,
          'pin_memory': True}

        print('=> Loading CIFAR10...')

        self.trainLoader = torch.utils.data.DataLoader(
            datasets.CIFAR10(opt.dataDir, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[ 0.4915,  0.4822,  0.4466],std=[ 0.2464,  0.2428,  0.2607]),
                       ])),
             **kwargs)

        self.valLoader = torch.utils.data.DataLoader(
            datasets.CIFAR10(opt.dataDir, train=False,
              transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[ 0.4915,  0.4822,  0.4466],std=[ 0.2464,  0.2428,  0.2607])
                       ])),
          **kwargs)
        self.numClasses = 10

class CIFAR100():
    def __init__(self, opt):
        kwargs = {
          'num_workers': opt.workers,
          'batch_size' : opt.batchSize,
          'shuffle' : True,
          'pin_memory': True}

        print('=> Loading CIFAR100...')

        self.trainLoader = torch.utils.data.DataLoader(
            datasets.CIFAR100(opt.dataDir, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[ 0.5071,  0.4866,  0.4410],std=[ 0.2664,  0.2556,  0.2752])
                       ])),
             **kwargs)

        self.valLoader = torch.utils.data.DataLoader(
            datasets.CIFAR100(opt.dataDir, train=False,
              transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[ 0.5071,  0.4866,  0.4410], std=[ 0.2664,  0.2556,  0.2752])
                       ])),
          **kwargs)
        self.numClasses = 100

class TinyImagenet():
    def __init__(self, opt):
        kwargs = {
          'num_workers': opt.workers,
          'batch_size' : opt.batchSize,
          'shuffle' : True,
          'pin_memory': True}

        print('=> Loading TinyImagenet...')

        dataTransforms = {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[ 0.4802,  0.4481,  0.3975], std=[ 0.2763,  0.2684,  0.2813])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[ 0.4802,  0.4481,  0.3975], std=[ 0.2763,  0.2684,  0.2813])
            ])
        }
        dataDir = opt.dataDir+'/TinyImagenet/'
        dtsets = {x: datasets.ImageFolder(os.path.join(dataDir, x), dataTransforms[x]) for x in ['train', 'val']}
        self.trainLoader = torch.utils.data.DataLoader(dtsets["train"], **kwargs)
        self.valLoader = torch.utils.data.DataLoader(dtsets["val"], **kwargs)
        self.numClasses = 200

class Imagenet12():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batchSize,
		  'shuffle' : True,
		  'pin_memory': True}

        print('=> Loading Imagenet...')

		dataTransforms = {
			'train': transforms.Compose([
				transforms.RandomResizedCrop(opt.inpsize),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]),
			'val': transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(opt.inpsize),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		}
		dataDir = opt.dataDir+'/Imagenet12/'
		dtsets = {x: datasets.ImageFolder(os.path.join(dataDir, x), dataTransforms[x]) for x in ['train', 'val']}
		self.trainLoader = torch.utils.data.DataLoader(dtsets["train"], **kwargs)
		self.valLoader = torch.utils.data.DataLoader(dtsets["val"], **kwargs)
    self.numClasses = 1000
