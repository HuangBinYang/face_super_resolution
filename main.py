import torch

from networks.generator import Generator
from networks.discriminator import Discriminator
from loss import PerceptualLoss
import torch.nn as nn


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super().__init__()
        self.extracted_layers = extracted_layers
        self.submodule: nn.Module = submodule

    def forward(self, x):
        outputs = []
        for name, module in self.submodule.named_modules():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]


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

    import ipdb; ipdb.set_trace()

