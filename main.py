import torch

from networks.generator import Generator
from networks.discriminator import Discriminator
from loss import PerceptualLoss
import torch.nn as nn
import torchvision.dataset as dset


if __name__ == "__main__":
    gen_net = Generator()
    images = torch.randn(5, 3, 128, 128)
    print(gen_net)
    output = gen_net(images)
    print(output.shape)

    dis_net = Discriminator()
    print(dis_net)
    print(dis_net(output).shape)
    perceptual_loss = PerceptualLoss()

    import ipdb

    ipdb.set_trace()
