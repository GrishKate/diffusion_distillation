import argparse
import torch
import os

from training import train
from dataloaders import make_ref_loader
from ddpm.ddpm import DDPM
from ddpm.unet import UNet
from ddpm_conditional_mnist.unet import ContextUnet
from ddpm_conditional_mnist.ddpm import ConditionalDDPM, ConditionalDDPMForward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="ddpm_conditional_mnist",
        choices=["ddpm_conditional_mnist", "ddpm", "ddmp_unconditional_celeba"],
    )
    parser.add_argument(
        "--pretrained_weights", type=str, default="/content/model_39.pth"
    )  # path to pretrained weights
    parser.add_argument(
        "--path_noise", type=str, default="/content/content/noise"
    )  # paths to dataset of noise-img pairs
    parser.add_argument(
        "--path_images", type=str, default="/content/content/images"
    )  # paths to dataset of noise-img pairs
    parser.add_argument(
        "--sigma_min", type=float, default=1e-4
    )  # 1e-4 for mnist, 1e-4 for unconditional net
    parser.add_argument(
        "--sigma_max", type=float, default=0.02
    )  # 2e-2 for mnist, 1e-2 for unconditional
    parser.add_argument(
        "--guide_w", type=float, default=0.0
    )  # works for conditional ddpm
    parser.add_argument("--drop_prob", type=float, default=0.1)  # for conditional ddpm
    parser.add_argument(
        "--T", type=float, default=400
    )  # 400 for ddpm_conditional_mnist, 1000 for unconditional
    parser.add_argument("--lr_netG", type=float, default=5e-6)  # 5e-6 for mnist
    parser.add_argument("--lr_mu_fake", type=float, default=5e-6)  # 5e-6 for mnist
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--training_steps", type=int, default=10)  # 10 enough for mnist
    parser.add_argument("--lambda_reg", type=float, default=0.5)
    parser.add_argument("--resume", type=str, default="")  # resume from checkpoint
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/content/drive/MyDrive/Colab Notebooks/diff_dist_cifar.pth",
    )  # where to save checkpoint
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("-f")  # to run in colab

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_name == "ddpm_conditional_mnist":
        fwd = ConditionalDDPMForward((args.sigma_min, args.sigma_max), args.T)
        forward_diffusion = fwd.forward
        checkpoint = torch.load(args.pretrained_weights, map_location=device)
        mu_real = ConditionalDDPM(
            ContextUnet(1, n_feat=128),
            betas=(args.sigma_min, args.sigma_max),
            n_T=args.T,
            device=device,
            drop_prob=args.drop_prob,
        )
        mu_real.load_state_dict(checkpoint)
        mu_fake = ConditionalDDPM(
            ContextUnet(1, n_feat=128),
            betas=(args.sigma_min, args.sigma_max),
            n_T=args.T,
            device=device,
            drop_prob=args.drop_prob,
        )
        mu_fake.load_state_dict(checkpoint)
        netG = ConditionalDDPM(
            ContextUnet(1, n_feat=128),
            betas=(args.sigma_min, args.sigma_max),
            n_T=args.T,
            device=device,
            drop_prob=args.drop_prob,
        )
        netG.load_state_dict(checkpoint)
    elif args.model_name == "ddpm":
        checkpoint = torch.load(args.pretrained_weights, map_location=device)
        mu_real = DDPM(
            UNet(args.n_T),
            n_steps=args.n_T,
            min_beta=args.sigma_min,
            max_beta=args.sigma_max,
            device=device,
        )
        mu_real.load_state_dict(checkpoint)
        forward_diffusion = mu_real.forward
        mu_fake = DDPM(
            UNet(args.n_T),
            n_steps=args.n_T,
            min_beta=args.sigma_min,
            max_beta=args.sigma_max,
            device=device,
        )
        mu_fake.load_state_dict(checkpoint)
        netG = DDPM(
            UNet(args.n_T),
            n_steps=args.n_T,
            min_beta=args.sigma_min,
            max_beta=args.sigma_max,
            device=device,
        )
        netG.load_state_dict(checkpoint)
    elif args.model_name == "ddmp_unconditional_celeba":
        from diffusers import UNet2DModel, DDPMScheduler

        args.n_T = 1000
        args.path_noise = "/content/GeneratedByDDPMCifarPairs/noise"
        args.path_images = "/content/GeneratedByDDPMCifarPairs/images"
        hugging_face_id = "google/ddpm-cifar10-32"
        pretrained_unet = UNet2DModel.from_pretrained(hugging_face_id)
        scheduler = DDPMScheduler.from_config(hugging_face_id)
        scheduler_params = scheduler.config
        imahe_shape = (
            pretrained_unet.config.in_channels,
            pretrained_unet.config.sample_size,
            pretrained_unet.config.sample_size,
        )

        mu_real = DDPM(
            network=pretrained_unet,
            device=device,
            image_chw=imahe_shape,
            scheduler_params=scheduler_params,
        )
        mu_fake = DDPM(
            network=UNet2DModel.from_pretrained(hugging_face_id),
            device=device,
            image_chw=imahe_shape,
            scheduler_params=scheduler_params,
        )
        netG = DDPM(
            network=UNet2DModel.from_pretrained(hugging_face_id),
            device=device,
            image_chw=imahe_shape,
            scheduler_params=scheduler_params,
        )
        forward_diffusion = mu_real.forward

        if os.path.exists(args.pretrained_weights):
            checkpoint = torch.load(args.pretrained_weights, map_location=device)
            mu_fake.load_state_dict(checkpoint)
            netG.load_state_dict(checkpoint)
    else:
        raise AttributeError("unknown model")

    ref_loader = make_ref_loader(args.path_noise, args.path_images, args.batch_size)
    train(args, forward_diffusion, mu_real, mu_fake, netG, ref_loader)


if __name__ == "__main__":
    main()
