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
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm, trange
from data import TrainDataset


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path", default="dataset/", type=str, help="Dataset path (default: dataset/)"
)

parser.add_argument(
    "--hr_size", default=256, type=int, help="High resolution size (default: 256)"
)
parser.add_argument(
    "--batch_size", default=16, type=int, help="Batch size (default: 256)"
)
parser.add_argument(
    "--scale_factor", default=4, type=int, help="Upscaling factor (default: 4)"
)
parser.add_argument(
    "--epochs", default=10, type=int, help="Number of epochs (default: 10)"
)
parser.add_argument(
    "--random_crop_size",
    default=96,
    type=int,
    help="Size of random crop during training (default: 96)",
)
parser.add_argument(
    "--gen_res_blocks",
    default=8,
    type=int,
    help="Generator residual blocks (default: 8)",
)

parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--cpu", dest="gpu", action="store_false")
parser.set_defaults(gpu=True)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and args.gpu) else "cpu"
    )

    dataset = TrainDataset(
        root=args.data_path,
        scale_factor=args.scale_factor,
        hr_size=args.hr_size,
        random_crop_size=args.random_crop_size,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    gen_net = Generator(
        num_res_blocks=args.gen_res_blocks, upscale_factor=args.scale_factor
    ).to(device)
    dis_net = Discriminator(hr_size=args.random_crop_size).to(device)
    print(f"Generator number of parameters: {count_parameters(gen_net)}")
    print(f"Discriminator number of parameters: {count_parameters(dis_net)}")
    perceptual_loss = PerceptualLoss(device=device)
    mse_loss = nn.MSELoss()
    lr = 0.0001
    beta1 = 0.9

    opt_gen = optim.Adam(gen_net.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_dis = optim.Adam(dis_net.parameters(), lr=lr, betas=(beta1, 0.999))

    for epoch in range(args.epochs):
        training_bar = tqdm(dataloader)
        for hr_sample, lr_sample in training_bar:
            hr_sample = hr_sample.to(device)
            lr_sample = lr_sample.to(device)

            sr_sample = gen_net(lr_sample)
            dis_net.zero_grad()
            sr_dis = dis_net(sr_sample).mean()
            hr_dis = dis_net(hr_sample).mean()
            d_loss = sr_dis + 1 - hr_dis
            d_loss.backward(retain_graph=True)
            opt_dis.step()

            gen_net.zero_grad()
            g_loss = (
                mse_loss(hr_sample, sr_sample)
                + perceptual_loss(hr_sample, sr_sample) * 0.006
                + (1 - sr_dis).mean() * 0.001
            )
            g_loss.backward()
            opt_gen.step()

            training_bar.set_description(
                f"Epoch: {epoch+1}/{args.epochs} / D_loss: {d_loss:.3f} / G_loss: {g_loss:.3f}"
            )
