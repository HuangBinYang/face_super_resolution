import torch

from networks.generator import Generator
from networks.discriminator import Discriminator

if __name__ == "__main__":
    gen_net = Generator()
    images = torch.randn(5, 3, 128, 128)
    print(gen_net)
    output = gen_net(images)
    print(output.shape)

    dis_net = Discriminator()
    print(dis_net)
    print(dis_net(output).shape)
