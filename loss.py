import torch
from torch import nn as nn
from torchvision.models.vgg import vgg19
import torchvision.transforms as transforms



device = "cpu"



class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class PerceptualNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True)
        features_slice = vgg.features[:35]
        perceptual_network = nn.Sequential(*features_slice).eval()
        for param in perceptual_network.parameters():
            param.requires_grad = False
        self.perceptual_network = perceptual_network

    def forward(self, inputs):
        return self.perceptual_network(inputs)


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1)
        self.transform = transforms.Normalize(self.cnn_normalization_mean, self.cnn_normalization_std)
        self.perceptual_network = PerceptualNetwork().to(device)
        self.mse_loss = nn.MSELoss()

    def forward(self, input_a, input_b):
        input_a = (input_a - self.cnn_normalization_mean) / self.cnn_normalization_std
        input_b = (input_b - self.cnn_normalization_mean) / self.cnn_normalization_std
        return self.mse_loss(
            self.perceptual_network(input_a), self.perceptual_network(input_b)
        )
