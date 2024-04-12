import torch
from distillation import denoising_loss, distribution_matching_loss, lpips


def save_checkpoints(epoch, save_dir, netG, mu_fake, optimG, optimMuFake):
    torch.save({'epoch': epoch,
                'netG': netG.state_dict(),
                'mu_fake': mu_fake.state_dict(),
                'optimG': optimG.state_dict(),
                'optimMuFake': optimMuFake.state_dict()}, save_dir)


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

    optimG = torch.optim.AdamW(netG.parameters(), lr=args.lr_netG,
                               betas=(args.beta_1, args.beta_2),
                               weight_decay=args.weight_decay)
    optimMuFake = torch.optim.AdamW(mu_fake.parameters(), lr=args.lr_mu_fake,
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
    mu_real.eval()
    mu_real.requires_grad_(False)
    cnt = -1
    for i in range(start_epoch, args.training_steps):
        for z_ref, y_ref in ref_loader:
            cnt += 1
            try:
                # generate images
                z_ref, y_ref = z_ref.to(device), y_ref.to(device)
                z = torch.randn_like(z_ref, device=device)
                x = netG.compute_x0(z, torch.full((z.shape[0],), netG.n_T - 1, device=device), args.guide_w)
                x_ref = netG.compute_x0(z_ref, torch.full((z.shape[0],), netG.n_T - 1, device=device), args.guide_w)
                # update generator
                loss_kl = distribution_matching_loss(mu_real, mu_fake, x, int(args.T * 0.02), int(args.T * 0.98),
                                                     args.batch_size, forward_diffusion, args.guide_w)
                loss_reg = lpips(x_ref, y_ref)
                loss = loss_kl + loss_reg * args.lambda_reg
                optimG.zero_grad()
                loss.backward()
                optimG.step()
                # Update fake score estimation model
                x = x.detach()
                noise = torch.randn_like(x)
                t = torch.randint(1, netG.n_T - 1, (x.shape[0],)).to(device)
                x_t = forward_diffusion(x, t, noise)
                # predict noise injected to x_t and calculate loss
                loss_denoise = denoising_loss(mu_fake.backward(x_t, t, args.guide_w), noise)
                optimMuFake.zero_grad()
                loss_denoise.backward()
                optimMuFake.step()
                print('step:', cnt, 'losses:', loss_kl.item(), loss_reg.item(), loss_denoise.item())

                if cnt % args.save_every == 0:
                    save_checkpoints(i, args.save_dir, netG, mu_fake, optimG, optimMuFake)
            except KeyboardInterrupt:
                save_checkpoints(i, args.save_dir, netG, mu_fake, optimG, optimMuFake)

    save_checkpoints(args.training_steps, args.save_dir, netG, mu_fake, optimG, optimMuFake)
    print('Finished training')
