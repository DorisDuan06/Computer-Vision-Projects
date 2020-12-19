import numpy as np
import scipy.io
import skimage.transform

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')


if __name__ == '__main__':
    squeezenet = models.squeezenet1_0(pretrained=True)
