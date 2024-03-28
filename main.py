import argparse
import torch

from training import train
from dataloaders import make_ref_loader
from ddpm.ddpm import DDPMForward
from ddpm.unet import NaiveUnet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ddpm')
    parser.add_argument('--pretrained_weights', type=str, default='')
    parser.add_argument('--path_noise', type=str, default='') # paths to dataset of noise-img pairs
    parser.add_argument('--path_images', type=str, default='') # paths to dataset of noise-img pairs
    parser.add_argument('--sigma_min', type=float, default=1e-4)#2e-3)
    parser.add_argument('--sigma_max', type=float, default=0.02)#80)
    parser.add_argument('--T', type=float, default=1000)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--training_steps', type=int, default=100)
    parser.add_argument('--lambda_reg', type=float, default=0.5)
    parser.add_argument('--resume', type=str, default='')  # resume from checkpoint
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--save_every', type=int, default=10)

    args = parser.parse_args()

    if args.model_name == 'ddpm':
        fwd = DDPMForward((args.sigma_min, args.sigma_max), args.T)
        forward_diffusion = fwd.forward
        mu_real = NaiveUnet(3, 3, n_feat=128)
        checkpoint = torch.load(args.pretrained_weights)
        mu_real.load_state_dict(checkpoint)
        mu_fake = NaiveUnet(3, 3, n_feat=128)
        netG = NaiveUnet(3, 3, n_feat=128)
    else:
        raise AttributeError('unknown model')

    ref_loader = make_ref_loader(args.path_noise, args.path_images, args.batch_size)
    train(args, forward_diffusion, mu_real, mu_fake, netG, ref_loader)


if __name__ == '__main__':
    main()
