from torch import nn as nn
import numpy as np
from .helpers import ResidualBlock, PixelShuffleBlock


class Generator(nn.Module):
    def __init__(self, num_res_blocks=16, upscale_factor=4):
        super(Generator, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.upscale_factor = upscale_factor
        power = int(np.log2(self.upscale_factor))
        assert 2**power == self.upscale_factor
        self.conv_before = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4
        )
        self.prelu_before = nn.PReLU()
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock() for _ in range(self.num_res_blocks)]
        )

        self.conv_after_1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn_after_1 = nn.BatchNorm2d(num_features=64)

        self.pixel_shuffle_blocks = nn.Sequential(
            *[PixelShuffleBlock() for _ in range(power)]
        )
        self.conv_after_2 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4
        )

    def forward(self, inputs):
        x = inputs
        x = self.conv_before(x)
        x = self.prelu_before(x)
        skip = x
        x = self.residual_blocks(x)
        x = self.conv_after_1(x)
        x = self.bn_after_1(x)
        x = x + skip
        x = self.pixel_shuffle_blocks(x)
        x = self.conv_after_2(x)
        return x
