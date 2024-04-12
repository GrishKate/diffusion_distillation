import torch
import os
from torch.utils.data import DataLoader, Dataset
import tqdm


def generate_noise_img_pairs(
    mu_real,
    path_noise,
    path_images,
    shape=(64, 3, 32, 32),
    batches_num=100,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    # generated dataset of noise-image pairs for reg loss
    # save noise and images as tensors with shape=shape
    assert path_noise != path_images
    mu_real.to(device)
    mu_real.eval()
    with torch.no_grad():
        for i in range(batches_num):
            print(i)
            noise = torch.randn(shape, device=device)
            images = mu_real.backward(noise)
            torch.save(noise.detach().cpu(), path_noise + "/{}.pt".format(i))
            torch.save(images.detach().cpu(), path_images + "/{}.pt".format(i))


def generate_noise_img_pairs_cifar(
    model,
    scheduler,
    path_noise,
    path_images,
    batch_size=64,
    batches_num=100,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batches_start_with=0,
):
    # generated dataset of noise-image pairs for reg loss
    # save noise and images as tensors with shape=(batch_size, 3, 32, 32)
    assert path_noise != path_images
    assert os.path.exists(path_noise) and os.path.exists(path_images)
    model.to(device)
    model.eval()
    torch.manual_seed(0)

    def make_paired_samples(bach_size):
        noisy_sample = torch.randn(
            bach_size,
            model.config.in_channels,
            model.config.sample_size,
            model.config.sample_size,
            device=device,
        )
        sample = noisy_sample.clone()
        for t in tqdm.tqdm(scheduler.timesteps):
            with torch.no_grad():
                residual = model(sample, t).sample
            sample = scheduler.step(residual, t, sample).prev_sample
        return noisy_sample, sample

    for i in range(batches_start_with, batches_num + batches_start_with):
        noise, images = make_paired_samples(bach_size=batch_size)
        torch.save(noise.detach().cpu(), path_noise + "/{}.pt".format(i))
        torch.save(images.detach().cpu(), path_images + "/{}.pt".format(i))


class MyDataset(Dataset):
    def __init__(self, path_noise, path_images, tf=None):
        self.noise = sorted(os.listdir(path_noise))
        self.images = sorted(os.listdir(path_images))
        self.bs = torch.load(os.path.join(path_noise, self.noise[0])).shape[0]
        assert len(self.noise) == len(self.images)
        self.path_noise = path_noise
        self.path_images = path_images

    def __getitem__(self, index):
        x = torch.load(os.path.join(self.path_noise, self.noise[index // self.bs]))[
            index % self.bs
        ]
        y = torch.load(os.path.join(self.path_images, self.images[index // self.bs]))[
            index % self.bs
        ]
        return x, y  # noise, image

    def __len__(self):
        length = len(self.noise) * self.bs
        return length


def make_ref_loader(path_noise, path_images, batch_size, tf=None):
    # input: paths to noise-image pairs
    # return: dataloader of noise-image pairs
    dataset = MyDataset(path_noise, path_images, tf)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    return loader
