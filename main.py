import torch

from networks.generator import Generator
from networks.discriminator import Discriminator
from loss import PerceptualLoss
import torch.nn as nn
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np


if __name__ == "__main__":
    # gen_net = Generator()
    # images = torch.randn(5, 3, 128, 128)
    # print(gen_net)
    # output = gen_net(images)
    # print(output.shape)
    #
    # dis_net = Discriminator()
    # print(dis_net)
    # print(dis_net(output).shape)
    # perceptual_loss = PerceptualLoss()
    dataroot = "dataset/"
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    batch = next(iter(dataloader))


    import ipdb

    ipdb.set_trace()
