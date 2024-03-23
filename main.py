import argparse
from training import train
from dataloaders import make_ref_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma_min', type=float, default=2e-3)
    parser.add_argument('--sigma_max', type=float, default=80)
    parser.add_argument('--n_bins', type=float, default=1000)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--training_steps', type=int, default=100000)
    parser.add_argument('--lambda_reg', type=float, default=0.5)
    parser.add_argument('--resume', type=str, default='') # resume from checkpoint
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--save_every', type=int, default=10)

    args = parser.parse_args()

    ref_loader = make_ref_loader()
    train(args, forward_diffusion, mu_real, mu_fake, netG, ref_loader)


if __name__ == '__main__':
    main()
