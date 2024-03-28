import torch
from distillation import denoising_loss, distribution_matching_loss, lpips


def save_checkpoints(epoch, save_dir, netG, mu_fake, optimG, optimMuFake):
    torch.save({'epoch': epoch,
                'netG': netG,
                'mu_fake': mu_fake,
                'optimG': optimG,
                'optimMuFake': optimMuFake}, save_dir)


def load_checkpoints(resume, netG, mu_fake, optimG, optimMuFake):
    checkpoint = torch.load(resume)
    netG.load_state_dict(checkpoint['netG'])
    mu_fake.load_state_dict(checkpoint['mu_fake'])
    optimG.load_state_dict(checkpoint['optimG'])
    optimMuFake.load_state_dict(checkpoint['optimMuFake'])
    epoch = checkpoint['epoch']
    return epoch, netG, mu_fake, optimG, optimMuFake


def train(args, forward_diffusion, mu_real, mu_fake, netG, ref_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # mu_real, mu_fake: denoising networks for real and fake distribution
    # netG : one-step generator

    # copy weights
    netG.load_state_dict(mu_real.state_dict())
    mu_fake.load_state_dict(mu_real.state_dict())

    optimG = torch.optim.AdamW(netG.params(), lr=args.lr,
                               betas=(args.beta_1, args.beta_2),
                               weight_decay=args.weight_decay)
    optimMuFake = torch.optim.AdamW(mu_fake.params(), lr=args.lr,
                                    betas=(args.beta_1, args.beta_2),
                                    weight_decay=args.weight_decay)
    start_epoch = 0
    if args.resume:
        start_epoch, netG, mu_fake, optimG, optimMuFake = load_checkpoints(args.resume,
                                                                           netG, mu_fake,
                                                                           optimG, optimMuFake)
    mu_real.to(device)
    mu_fake.to(device)
    netG.to(device)
    netG.train()
    mu_fake.train()
    for i in range(start_epoch, args.training_steps):
        try:
            # generate images
            z_ref, y_ref = ref_loader.next()
            z_ref, y_ref = z_ref.to(device), y_ref.to(device)
            z = torch.randn_like(z_ref, device=device)
            x = netG(z)
            x_ref = netG(z_ref)
            # update generator
            loss_kl = distribution_matching_loss(mu_real, mu_fake, x, 1, args.T,
                                                 args.batch_size, forward_diffusion)
            loss_reg = lpips(x_ref, y_ref)
            loss = loss_kl + loss_reg * args.lambda_reg
            optimG.zero_grad()
            loss.backward()
            optimG.step()
            # Update fake score estimation model
            t = torch.randint(1, args.T + 1, (x.shape[0],)).to(x.device)
            x_t = forward_diffusion(x.detach(), t)
            loss_denoise = denoising_loss(mu_fake(x_t, t), x.detach())
            optimMuFake.zero_grad()
            loss_denoise.backward()
            optimMuFake.step()

            if i % args.save_every == 0:
                save_checkpoints(i, args.save_dir, netG, mu_fake, optimG, optimMuFake)
        except KeyboardInterrupt:
            save_checkpoints(i, args.save_dir, netG, mu_fake, optimG, optimMuFake)

    save_checkpoints(args.training_steps, args.save_dir, netG, mu_fake, optimG, optimMuFake)
    print('Finished training')
