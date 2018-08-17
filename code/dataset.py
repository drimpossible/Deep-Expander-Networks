from __future__ import print_function
from __future__ import division

from pathlib import Path

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from data import datasetList


def getDataloader(args):
    dataloader = getattr(datasetList, args.dataset)(args)
    return dataloader.trainLoader, dataloader.valLoader, dataloader.numClasses
