'''
负责：

下载 / 读取 CIFAR-10、SVHN
定义 transform
返回 train/val/test dataset
'''

import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split


def get_default_transform():
    return transforms.ToTensor()


def load_cifar10(root="data", transform=None, val_ratio=0.1, seed=42):
    """
    加载 CIFAR-10，并从训练集中分出验证集
    
    Returns:
        train_set, val_set, test_set
    """
    if transform is None:
        transform = get_default_transform()

    full_train = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform
    )
    test_set = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=transform
    )
    
    # 从训练集分出验证集
    train_size = int(len(full_train) * (1 - val_ratio))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_set, val_set, test_set


def load_svhn(root="data", transform=None, val_ratio=0.1, seed=42):
    """
    加载 SVHN，并从训练集中分出验证集
    
    Returns:
        train_set, val_set, test_set
    """
    if transform is None:
        transform = get_default_transform()

    full_train = datasets.SVHN(
        root=root,
        split="train",
        download=True,
        transform=transform
    )
    test_set = datasets.SVHN(
        root=root,
        split="test",
        download=True,
        transform=transform
    )
    
    # 从训练集分出验证集
    train_size = int(len(full_train) * (1 - val_ratio))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_set, val_set, test_set


def load_all_datasets(root="data", transform=None, val_ratio=0.1, seed=42):
    """
    加载所有数据集
    
    Returns:
        cifar_train, cifar_val, cifar_test, svhn_train, svhn_val, svhn_test
    """
    cifar_train, cifar_val, cifar_test = load_cifar10(root=root, transform=transform, val_ratio=val_ratio, seed=seed)
    svhn_train, svhn_val, svhn_test = load_svhn(root=root, transform=transform, val_ratio=val_ratio, seed=seed)
    return cifar_train, cifar_val, cifar_test, svhn_train, svhn_val, svhn_test