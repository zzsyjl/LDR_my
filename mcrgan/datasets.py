# from __future__ import print_function
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import random
import glob
import PIL
import torchvision.datasets as datasets
import torchvision.transforms.functional as F
from torch_mimicry.datasets.data_utils import load_dataset
from typing import Any, Tuple


class MyAffineTransform:
    """Transform by one of the ways."""
    '''Choice:[angle, scale]'''

    def __init__(self, choices):
        self.choices = choices

    def __call__(self, x):
        choice = random.choice(self.choices)
        x = F.affine(x, angle=choice[0], scale=choice[1], translate=[0,0], shear=0)
        return x


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


class celeba_dataset(data.Dataset):

    def __init__(self, root, size):

        self.files = sorted(glob.glob(f"{root}/*.jpg"))

        transforms_list = [
            transforms.CenterCrop(
                178),  # Because each image is size (178, 218) spatially.
            transforms.Resize(size)
        ]
        transforms_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]

        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):

        X = PIL.Image.open(self.files[item])

        return self.transform(X), 0

def create_split_subsets(dataset, num_splits):
	total_num_classes = len(dataset.classes)
	classes_per_split = total_num_classes // num_splits
	subsets = []
	for i in range(0, num_splits):
		indices = torch.nonzero(torch.tensor(dataset.targets)//classes_per_split == i , as_tuple=True)[0]
		subset = torch.utils.data.Subset(dataset, indices)
		subsets.append(subset)
	return subsets

def get_dataloaders(data_name, num_splits, root, batch_size, num_workers):

    if data_name in ["lsun_bedroom_128", "stl10_48"]:
        dataset = load_dataset(root=root, name=data_name)

    elif data_name == 'cifar10':
        transform = transforms.Compose(
                    [
                    # transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)])
        dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)

    elif data_name == 'celeba':
        dataset = celeba_dataset(root=root, size=128)

    elif data_name == 'mnist':

        transform = transforms.Compose(
                    [transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)])
        dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)

    elif data_name == 'TMNIST':
        transform = transforms.Compose(
                    [transforms.Resize(32),
                    transforms.ToTensor(),
                    MyAffineTransform(choices=[[0, 1], [0, 1.5], [0, 0.5], [-45, 1], [45, 1]]),
                    transforms.Normalize(0.5, 0.5)])
        dataset = datasets.MNIST(root=root, train=True,
                                    download=True, transform=transform)

    elif data_name == 'imagenet_128':
        dataset = datasets.ImageFolder(root,
                                       transform=transforms.Compose([
                                         transforms.CenterCrop(224),
                                         transforms.Resize(size=(128, 128)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                         transforms.Lambda(lambda x: x + torch.rand_like(x) / 128)
                                       ]))

    else:
        raise ValueError()

    subsets = create_split_subsets(dataset, num_splits)
    dataloaders = []
    for subset in subsets:
        dataloader = torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        dataloaders.append(dataloader)
    return dataloaders

def get_dataloader(data_name, root, batch_size, num_workers, dataset_size=1.0):

    if data_name in ["lsun_bedroom_128", "cifar10", "stl10_48"]:
        dataset = load_dataset(root=root, name=data_name)

    elif data_name == 'celeba':
        dataset = celeba_dataset(root=root, size=128)

    elif data_name == 'mnist':

        transform = transforms.Compose(
                    [transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)])
        dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)

    elif data_name == 'TMNIST':
        transform = transforms.Compose(
                    [transforms.Resize(32),
                    transforms.ToTensor(),
                    MyAffineTransform(choices=[[0, 1], [0, 1.5], [0, 0.5], [-45, 1], [45, 1]]),
                    transforms.Normalize(0.5, 0.5)])
        dataset = datasets.MNIST(root=root, train=True,
                                    download=True, transform=transform)

    elif data_name == 'imagenet_128':
        dataset = datasets.ImageFolder(root,
                                       transform=transforms.Compose([
                                         transforms.CenterCrop(224),
                                         transforms.Resize(size=(128, 128)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                         transforms.Lambda(lambda x: x + torch.rand_like(x) / 128)
                                       ]))

    else:
        raise ValueError()
    if dataset_size < 1.0:
        dataset = torch.utils.data.Subset(dataset, range(0, int(len(dataset)*dataset_size)))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader, dataset
