import numpy as np

import torch
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import os

def get_train_loader(data_dir,
                     batch_size,
                     shuffle=True,
                     num_workers=4,
                     pin_memory=True):

    # define transforms
    trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # load dataset
    dataset = datasets.CIFAR100(root=data_dir,
                                transform=trans,
                                download=True,
                                train=True)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader


def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=True):

    # define transforms
    trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # load dataset
    dataset = datasets.CIFAR100(
        data_dir, train=False, download=True, transform=trans
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

def get_IN_train_loader(data_dir,
                        batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True):
    # define transforms
    trans = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # load dataset
    dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=trans
    )

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader


def get_IN_test_loader(data_dir,
                       batch_size,
                       num_workers=4,
                       pin_memory=True):
    # define transforms
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # load dataset
    dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=trans
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

