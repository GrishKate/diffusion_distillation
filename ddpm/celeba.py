import os
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, Resize


class CelebADataset(Dataset):
    def __init__(self, image_label_mapping, transform=None):
        """
        Args:
            image_label_mapping (dict): A dictionary mapping image paths to labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_label_mapping = image_label_mapping
        self.transform = transform
        self.image_paths = list(image_label_mapping.keys())

    def __len__(self):
        return len(self.image_label_mapping)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.image_label_mapping[image_path]

        if self.transform:
            image = self.transform(image)

        return image


def my_norm(x):
    return x * 2 - 1


def my_denormalize(x):
    return torch.clamp((x + 1) / 2, min=0, max=1)


transform = Compose([
    Resize((32, 32)),
    ToTensor(),
    Lambda(my_norm),
])


def make_loader(batch_size):
    # specify directories
    img_dir = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'
    attributes_file = '/kaggle/input/celeba-dataset/list_attr_celeba.csv'
    partition_file = '/kaggle/input/celeba-dataset/list_eval_partition.csv'
    attr_df = pd.read_csv(attributes_file)
    partition_df = pd.read_csv(partition_file)
    merged_df = pd.merge(attr_df[['image_id', 'Male']], partition_df, on="image_id")
    train_df = merged_df[merged_df['partition'] == 0]
    image_label_mapping = {os.path.join(img_dir, row['image_id']): 1 if row['Male'] > 0 else 0 for _, row in
                           train_df.iterrows()}
    image_paths = list(image_label_mapping.keys())
    labels = list(image_label_mapping.values())
    mapping = dict(zip(image_paths, labels))
    dataset = CelebADataset(mapping, transform=transform)
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4)
    return loader
