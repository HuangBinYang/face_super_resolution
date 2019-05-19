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
from data import VisDataset
import torchvision.utils as vutils
import os
from collections import defaultdict
import yaml


parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_path",
    default="example_images/",
    type=str,
    help="Test dataset path (default: example_images)",
)

parser.add_argument(
    "--from_pretrained_gen", default="pretrained_models/generator.pth", help="Load models and start training from them"
)

parser.add_argument("--hr_size", default=256, help="High resolution size")
parser.add_argument(
    "--vis_batch_size", default=8, type=int, help="Batch size for visualisation"
)
parser.add_argument("--iterations", default=8, type=int, help="Number of batches")
parser.add_argument("--scale_factor", default=4, help="Scale factor of the generator")
parser.add_argument(
    "--gen_res_blocks", default=8, help="Number of residual blocks of the generator"
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
    compare_dir = f"compare/{now}"
    os.makedirs(compare_dir, exist_ok=True)

    test_dataset = VisDataset(
        root=args.data_path, scale_factor=args.scale_factor, hr_size=args.hr_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.vis_batch_size
    )

    gen_net = Generator(
        num_res_blocks=args.gen_res_blocks, upscale_factor=args.scale_factor
    ).to(device)

    gen_path = args.from_pretrained_gen
    if gen_path:
        gen_net.load_state_dict(torch.load(gen_path))

    test_iter = iter(test_dataloader)

    for i, (hr_test, lr_test, simple_sr_test) in zip(range(args.iterations), test_iter):
        gan_sr_test = gen_net(lr_test.to(device)).cpu()
        stacked = torch.cat([hr_test, simple_sr_test, gan_sr_test])
        vutils.save_image(
            stacked, f"{compare_dir}/{i}.png", normalize=True, nrow=args.vis_batch_size
        )
