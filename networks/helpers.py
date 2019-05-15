from torch import nn as nn


class PixelShuffleBlock(nn.Module):
    def __init__(self):
        super(PixelShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()
        self.net = nn.Sequential(self.conv, self.pixel_shuffle, self.prelu)

    def forward(self, inputs):
        return self.net(inputs)


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=64)

        self.net = nn.Sequential(self.conv1, self.bn1, self.prelu, self.conv2, self.bn2)

    def forward(self, inputs):
        return self.net(inputs) + inputs
