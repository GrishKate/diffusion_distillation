import torch
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image


def generate_noise_img_pairs(mu_real, path_noise, path_images,
                             shape=(64, 3, 32, 32), batches_num=100, denormalize=None):
    # generated dataset of noise-image pairs for reg loss
    # save noise and images as tensors with shape=shape
    '''
    if denormalize is None:
        denormalize = transforms.Compose([transforms.Normalize((0.0, 0.0, 0.0), (1/0.5, 1/0.5, 1/0.5)),
                                          transforms.Normalize((-0.5, -0.5, -0.5), (1.0, 1.0, 1.0))])
    '''
    assert path_noise != path_images
    for i in range(batches_num):
        noise = torch.randn(shape)
        images = mu_real(noise)
        torch.save(noise, path_noise + '/{}.pt'.format(i))
        torch.save(images, path_images + '/{}.pt'.format(i))


class MyDataset(Dataset):
    def __init__(self, path_noise, path_images, tf=None):
        self.noise = sorted(os.listdir(path_noise))
        self.images = sorted(os.listdir(path_images))
        self.bs = torch.load(self.noise[0]).shape[0]
        assert len(self.noise) == len(self.images)

    def __getitem__(self, index):
        x = torch.load(self.noise[index // self.bs])[index % self.bs]
        y = torch.load(self.images[index // self.bs])[index % self.bs]
        return x, y

    def __len__(self):
        return len(self.noise)


def make_ref_loader(path_noise, path_images, batch_size, tf=None):  # paths to noise-image pairs
    # return: dataloader of noise-image pairs
    dataset = MyDataset(path_noise, path_images, tf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader


'''
class MyDataset(Dataset):
    def __init__(self, path_noise, path_images, tf=None):
        self.noise = sorted(os.listdir(path_noise))
        self.images = sorted(os.listdir(path_images))
        assert len(self.noise) == len(self.images)
        self.transform = tf
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        x = Image.open(self.noise[index])
        y = Image.open(self.images[index])
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

    def __len__(self):
        return len(self.noise)
'''
