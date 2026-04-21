"""
负责：

下载 / 读取 CIFAR-10、SVHN
定义 transform
返回 train/val/test dataset
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset


# ================= 数据预处理 =================
def get_cifar10_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),  # 先填充4像素，再随机裁剪回32×32，增加平移不变性
            transforms.RandomHorizontalFlip(),  # 随机水平翻转，增加数据多样性
            transforms.ToTensor(),  # 转为张量，像素值从[0,255]归一化到[0,1]
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize(mean, std) 标准化到均值0、标准差1，加速训练收敛
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    return train_transform, test_transform


def get_svhn_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(10),  # 随机旋转±10度
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机调整亮度和对比度±20%
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ]
    )
    return train_transform, test_transform

def split_indices(n, val_ratio=0.1, seed=42):
    """
    将样本索引随机划分为 train / val
    """
    g = torch.Generator().manual_seed(seed) # 创建随机数生成器
    indices = torch.randperm(n, generator=g).tolist() # torch.randperm(n)表示生成一个 0 ~ n-1 的随机排列。
    train_size = int(n * (1 - val_ratio)) # 设置训练集的大小
    train_indices = indices[:train_size] # 然后根据索引indices切分
    val_indices = indices[train_size:]
    return train_indices, val_indices

def load_cifar10(root="data", val_ratio=0.1, seed=42,
                 train_transform=None, test_transform=None):
    """
    加载 CIFAR-10，并从训练集中分出验证集
    Returns:
        train_set, val_set, test_set
    """
    if train_transform is None or test_transform is None:
        train_transform, test_transform = get_cifar10_transforms()
    # 先下载一次基础数据，用于切分索引
    base_train = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=None
    )
    test_set = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=test_transform
    )
    train_indices, val_indices = split_indices(len(base_train), val_ratio=val_ratio, seed=seed)
    # 分别构造训练集和验证集，使用不同 transform
    train_base = datasets.CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=train_transform
    )
    val_base = datasets.CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=test_transform
    )
    train_set = Subset(train_base, train_indices)
    val_set = Subset(val_base, val_indices)
    return train_set, val_set, test_set
def load_svhn(root="data", val_ratio=0.1, seed=42,
              train_transform=None, test_transform=None):
    """
    加载 SVHN，并从训练集中分出验证集
    Returns:
        train_set, val_set, test_set
    """
    if train_transform is None or test_transform is None:
        train_transform, test_transform = get_svhn_transforms()
    base_train = datasets.SVHN(
        root=root,
        split="train",
        download=True,
        transform=None
    )
    test_set = datasets.SVHN(
        root=root,
        split="test",
        download=True,
        transform=test_transform
    )
    train_indices, val_indices = split_indices(len(base_train), val_ratio=val_ratio, seed=seed)
    train_base = datasets.SVHN(
        root=root,
        split="train",
        download=False,
        transform=train_transform
    )
    val_base = datasets.SVHN(
        root=root,
        split="train",
        download=False,
        transform=test_transform
    )
    train_set = Subset(train_base, train_indices)
    val_set = Subset(val_base, val_indices)
    return train_set, val_set, test_set

def load_all_datasets(root="data", val_ratio=0.1, seed=42):
    """
    加载所有数据集
    Returns:
        cifar_train, cifar_val, cifar_test, svhn_train, svhn_val, svhn_test
    """
    cifar_train, cifar_val, cifar_test = load_cifar10(
        root=root, val_ratio=val_ratio, seed=seed
    )
    svhn_train, svhn_val, svhn_test = load_svhn(
        root=root, val_ratio=val_ratio, seed=seed
    )
    return cifar_train, cifar_val, cifar_test, svhn_train, svhn_val, svhn_test