import torch

from networks.generator import Generator

if __name__ == "__main__":
    gen_net = Generator()
    images = torch.randn(5, 3, 128, 128)
    print(gen_net)
    print(gen_net(images).shape)
