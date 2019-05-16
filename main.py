from networks.generator import Generator
from networks.discriminator import Discriminator
from loss import PerceptualLoss, calc_grad_pen
import torch.nn as nn
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import datetime
from tqdm import tqdm, trange
from data import TrainDataset, TestDataset
import torchvision.utils as vutils
import os
from collections import defaultdict
import yaml

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_data_path",
    default="dataset/",
    type=str,
    help="Training ataset path (default: data/train_dataset/)",
)
parser.add_argument(
    "--test_data_path",
    default="dataset/",
    type=str,
    help="Test dataset path (default: data/test_dataset/)",
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
    "--start_discriminator",
    default=20,
    type=int,
    help="Number of epochs passed to start discriminator (default: 4)",
)
parser.add_argument(
    "--epochs", default=100, type=int, help="Number of epochs (default: 10)"
)
parser.add_argument(
    "--critic_iters", default=5, type=int, help="Number of epochs (default: 10)"
)
parser.add_argument(
    "--save_every",
    default=300,
    type=int,
    help="Save every ... iterations in epoch (default: 10)",
)
parser.add_argument(
    "--wgan_gp_lambda",
    default=10.0,
    type=float,
    help="WGAN-GP coefficient (default: 10)",
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
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    results_dir = f"results/{now}"
    models_dir = f"models/{now}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    train_dataset = TrainDataset(
        root=args.train_data_path,
        scale_factor=args.scale_factor,
        hr_size=args.hr_size,
        random_crop_size=args.random_crop_size,
    )
    test_dataset = TestDataset(
        root=args.test_data_path, scale_factor=args.scale_factor, hr_size=args.hr_size
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

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

    hr_test, _ = next(iter(test_dataloader))
    vutils.save_image(hr_test, f"{results_dir}/hr.png", normalize=True)
    for epoch in range(args.epochs):
        training_bar = tqdm(train_dataloader)
        stats = defaultdict(float)
        for i, (hr_sample, lr_sample) in enumerate(training_bar, 1):
            gen_net = gen_net.train()

            if epoch >= args.start_discriminator:
                hr_sample = hr_sample.to(device)
                lr_sample = lr_sample.to(device)
                sr_sample = gen_net(lr_sample)
                dis_net.zero_grad()
                sr_dis = dis_net(sr_sample).mean()
                hr_dis = dis_net(hr_sample).mean()

                gradient_penalty = (
                    calc_grad_pen(dis_net, hr_sample, sr_sample, device)
                    * args.wgan_gp_lambda
                )

                d_loss = sr_dis + 1 - hr_dis
                d_loss_total = d_loss + gradient_penalty
                d_loss_total.backward(retain_graph=True)
                opt_dis.step()
            else:
                hr_sample = hr_sample.to(device)
                lr_sample = lr_sample.to(device)
                d_loss = 0
                gradient_penalty = 0

            gen_net.zero_grad()
            sr_sample = gen_net(lr_sample)
            sr_dis = dis_net(sr_sample).mean()

            mse_part = mse_loss(hr_sample, sr_sample)
            perceptual_part = perceptual_loss(hr_sample, sr_sample) * 0.012
            if epoch >= args.start_discriminator:
                gan_part = (1 - sr_dis).mean() * 0.001
            else:
                gan_part = 0
            g_loss = mse_loss(hr_sample, sr_sample) + perceptual_part + gan_part
            g_loss.backward()
            opt_gen.step()

            stats["denom"] += 1

            stats["d_loss_sum"] += d_loss
            stats["d_loss"] = stats["d_loss_sum"] / stats["denom"]

            stats["gradient_penalty_sum"] += gradient_penalty
            stats["gradient_penalty"] = (
                stats["gradient_penalty_sum"] / stats["denom"]
            )

            stats["mse_part_sum"] += mse_part
            stats["mse_part"] = stats["mse_part_sum"] / stats["denom"]

            stats["perceptual_part_sum"] += perceptual_part
            stats["perceptual_part"] = stats["perceptual_part_sum"] / stats["denom"]

            stats["gan_part_sum"] += gan_part
            stats["gan_part"] = stats["gan_part_sum"] / stats["denom"]

            training_bar.set_description(
                f"Epoch: {epoch+1}/{args.epochs} "
                f"/ D_loss: {stats['d_loss']:.3f},{stats['gradient_penalty']:.3f} "
                f"/ G_loss: {stats['mse_part']:.3f},{stats['perceptual_part']:.3f},{stats['gan_part']:.3f}"
            )
            if i % args.save_every == 0:
                _, lr_test = next(iter(test_dataloader))
                gen_net = gen_net.eval()
                sr_test = gen_net(lr_test.to(device)).cpu()
                vutils.save_image(
                    sr_test, f"{results_dir}/sr_{epoch}_{i}.png", normalize=True
                )
                del sr_test
                torch.cuda.empty_cache()
        with open(f'{models_dir}/trainings_stats_{epoch}.yml', 'w') as f:
            yaml.dump(stats, f)
        torch.save(gen_net.state_dict(), f"{models_dir}/gen_epoch_{epoch}.pth")
        torch.save(dis_net.state_dict(), f"{models_dir}/dis_epoch_{epoch}.pth")
