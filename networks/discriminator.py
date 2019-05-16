import torch.nn as nn
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, inputs):
        x = inputs
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, hr_size=512):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        tmp = hr_size // 8 
        self.leaky_relu = nn.LeakyReLU(0.2)
        params = [
            (64, 64, 2),
            (64, 128, 1),
            (128, 128, 2),
            (128, 256, 1),
            (256, 256, 2),
            (256, 512, 1),
            (512, 512, 2),
        ]
        self.blocks_list = nn.ModuleList()
        for in_c, out_c, stride in params:
            self.blocks_list.append(Block(in_c, out_c, stride))
        self.blocks = nn.Sequential(*self.blocks_list)
        self.dense1 = nn.Linear(512 * tmp * tmp, 1024)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = inputs
        x = self.conv(x)
        x = self.leaky_relu(x)
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.leaky_relu2(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x
