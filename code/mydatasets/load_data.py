import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import utils

class LoadCIFAR10():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10(opt.data_dir, train=True, download=True,
					transform=transforms.Compose([
						transforms.RandomCrop(32, padding=4),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
std=[x/255.0 for x in [63.0, 62.1, 66.7]])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10(opt.data_dir, train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
std=[x/255.0 for x in [63.0, 62.1, 66.7]])
					   ])),
		  **kwargs)

class LoadCIFAR100():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR100(opt.data_dir, train=True, download=True,
					transform=transforms.Compose([
						transforms.RandomCrop(32, padding=4),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize(mean=[x/255.0 for x in [129.3, 124.1, 112.4]],
std=[x/255.0 for x in [68.2, 65.4, 70.4]])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			datasets.CIFAR100(opt.data_dir, train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[x/255.0 for x in [129.3, 124.1, 112.4]],
std=[x/255.0 for x in [68.2, 65.4, 70.4]])
					   ])),
		  **kwargs)

class LoadImagenet12():
	def __init__(self, opt):
		kwargstrain = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : False,
		  'pin_memory': True}
		kwargstest = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}
		data_transforms = {
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
		data_dir = opt.data_dir
		dtsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
		self.train_loader = torch.utils.data.DataLoader(dtsets["train"], **kwargstrain)
		self.val_loader = torch.utils.data.DataLoader(dtsets["val"], **kwargstest)
