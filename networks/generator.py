from torch import nn as nn

from .helpers import ResidualBlock, PixelShuffleBlock


class Generator(nn.Module):
    def __init__(self, num_res_blocks=16):
        super(Generator, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_before_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4
        )
        self.prelu1 = nn.PReLU()
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock() for _ in range(self.num_res_blocks)]
        )

        self.conv_a1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn_a1 = nn.BatchNorm2d(num_features=64)

        self.pixel_shuffle_blocks = nn.Sequential(
            PixelShuffleBlock(), PixelShuffleBlock()
        )
        self.conv_a2 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4
        )

    def forward(self, inputs):
        x = inputs
        x = self.conv_before_1(x)
        x = self.prelu1(x)
        skip = x
        x = self.residual_blocks(x)
        x = self.conv_a1(x)
        x = self.bn_a1(x)
        x = x + skip
        x = self.pixel_shuffle_blocks(x)
        x = self.conv_a2(x)
        return x
