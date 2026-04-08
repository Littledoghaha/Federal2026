"""
“统一流程层 / 业务层”

负责为不同训练场景准备数据：
- 集中式学习（Centralized Learning）
- 联邦学习（Federated Learning）
- 持续学习（Continual Learning）
- 联邦持续学习（Federated Continual Learning, 预留）

主要职责：
1. 加载原始数据集
2. 根据不同实验场景进行划分
3. 构造子数据集（subset）
4. 构造 DataLoader
5. 返回训练流程直接可用的数据结构

整体流程：
原始数据 -> 划分/任务构造 -> subset -> dataloader -> 打包返回

这是面向“训练场景”的统一数据准备入口。
"""

from datasets.cifar_svhn import load_cifar10, load_svhn, load_all_datasets
from datasets.data_split import iid_split_indices, build_client_subsets
from client_dataloader import build_client_loaders, build_test_loader, build_train_loader
import numpy as np
from torch.utils.data import Subset


def prepare_centralized_dataloaders(dataset="cifar10", batch_size=64, test_batch_size=128, seed=42, num_workers=0, val_ratio=0.1):
    """
    集中式学习的数据准备入口
    流程：
    原始数据 -> train/val/test -> dataloader
    返回：
        train_loader, val_loader, test_loader
    """
    if dataset == "cifar10":
        train_set, val_set, test_set = load_cifar10(val_ratio=val_ratio, seed=seed)
    elif dataset == "svhn":
        train_set, val_set, test_set = load_svhn(val_ratio=val_ratio, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    train_loader = build_train_loader(train_set, batch_size=batch_size, num_workers=num_workers)
    val_loader = build_test_loader(val_set, batch_size=test_batch_size, num_workers=num_workers)
    test_loader = build_test_loader(test_set, batch_size=test_batch_size, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def prepare_federated_dataloaders(num_clients=3, train_batch_size=64, test_batch_size=128, seed=42, num_workers=0, val_ratio=0.1, root="data"):
    """
    联邦学习的数据准备入口
    流程：
    原始数据 -> 客户端划分 -> client subsets -> client dataloaders -> 打包返回
    返回：
        {
            "cifar": {
                "train_dataset": ...,
                "val_dataset": ...,
                "test_dataset": ...,
                "client_indices": ...,
                "client_datasets": ...,
                "client_loaders": ...,
                "val_loader": ...,
                "test_loader": ...,
            },
            "svhn": {...}
        }
    """
    cifar_train, cifar_val, cifar_test, svhn_train, svhn_val, svhn_test = load_all_datasets(root=root, val_ratio=val_ratio, seed=seed)
    
    cifar_client_indices = iid_split_indices(cifar_train, num_clients=num_clients, seed=seed)
    svhn_client_indices = iid_split_indices(svhn_train, num_clients=num_clients, seed=seed)
    
    cifar_client_datasets = build_client_subsets(cifar_train, cifar_client_indices)
    svhn_client_datasets = build_client_subsets(svhn_train, svhn_client_indices)
    
    cifar_client_loaders = build_client_loaders(cifar_client_datasets, batch_size=train_batch_size, num_workers=num_workers)
    svhn_client_loaders = build_client_loaders(svhn_client_datasets, batch_size=train_batch_size, num_workers=num_workers)
    
    cifar_val_loader = build_test_loader(cifar_val, batch_size=test_batch_size, num_workers=num_workers)
    cifar_test_loader = build_test_loader(cifar_test, batch_size=test_batch_size, num_workers=num_workers)
    svhn_val_loader = build_test_loader(svhn_val, batch_size=test_batch_size, num_workers=num_workers)
    svhn_test_loader = build_test_loader(svhn_test, batch_size=test_batch_size, num_workers=num_workers)
    
    return {
        "cifar": {"train_dataset": cifar_train, "val_dataset": cifar_val, "test_dataset": cifar_test, "client_indices": cifar_client_indices, "client_datasets": cifar_client_datasets, "client_loaders": cifar_client_loaders, "val_loader": cifar_val_loader, "test_loader": cifar_test_loader},
        "svhn": {"train_dataset": svhn_train, "val_dataset": svhn_val, "test_dataset": svhn_test, "client_indices": svhn_client_indices, "client_datasets": svhn_client_datasets, "client_loaders": svhn_client_loaders, "val_loader": svhn_val_loader, "test_loader": svhn_test_loader},
    }


def prepare_continual_dataloaders(dataset="cifar10", batch_size=64, test_batch_size=128, seed=42, num_workers=0, val_ratio=0.1, num_tasks=2, task_classes=None):
    """
    持续学习的数据准备入口
    流程：
    原始数据 -> 按类别划分任务 -> 每个任务构造 train/val/test -> dataloader -> 打包返回
    说明：
    - 支持手动指定 task_classes
    - 若 task_classes=None，则按类别自动均匀切分任务
    返回：
        {
            "task0": {
                "train_dataset": ...,
                "val_dataset": ...,
                "test_dataset": ...,
                "train_loader": ...,
                "val_loader": ...,
                "test_loader": ...,
            },
            ...
        }
    """
    if dataset == "cifar10":
        train_set, val_set, test_set = load_cifar10(val_ratio=val_ratio, seed=seed)
        num_classes = 10
    elif dataset == "svhn":
        train_set, val_set, test_set = load_svhn(val_ratio=val_ratio, seed=seed)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # 自动生成任务类别划分
    if task_classes is None:
        classes_per_task = num_classes // num_tasks
        task_classes = [
            list(range(i * classes_per_task, (i + 1) * classes_per_task))
            for i in range(num_tasks)
        ]
    
    def filter_by_classes(dataset, classes):
        """
        从数据集中筛选指定类别的样本，并返回一个新的 Subset。

        注意：
        - 如果 dataset 本身就是 Subset，那么返回的索引必须是“相对当前 Subset 的索引”
        - 不能直接使用原始大数据集的绝对索引，否则会导致索引越界
        """
        indices = []

        if isinstance(dataset, Subset):
            # dataset 是某个原始数据集的子集
            # dataset.indices 是当前子集映射到原始数据集的索引列表
            base_targets = np.array(dataset.dataset.targets)

            # new_idx：当前 subset 内部的相对索引
            # original_idx：原始大数据集中的绝对索引
            for new_idx, original_idx in enumerate(dataset.indices):
                label = base_targets[original_idx]
                if label in classes:
                    indices.append(new_idx)
        else:
            # dataset 不是 Subset，直接遍历它自己的 targets
            labels = np.array(dataset.targets)
            for idx, label in enumerate(labels):
                if label in classes:
                    indices.append(idx)

        return Subset(dataset, indices)
    
    tasks = {}
    for task_id, classes in enumerate(task_classes):
        task_train = filter_by_classes(train_set, classes)
        task_val = filter_by_classes(val_set, classes)
        task_test = filter_by_classes(test_set, classes)
        
        train_loader = build_train_loader(task_train, batch_size=batch_size, num_workers=num_workers)
        val_loader = build_test_loader(task_val, batch_size=test_batch_size, num_workers=num_workers)
        test_loader = build_test_loader(task_test, batch_size=test_batch_size, num_workers=num_workers)
        
        tasks[f"task{task_id}"] = {
            "train_dataset": task_train,
            "val_dataset": task_val,
            "test_dataset": task_test,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
        }
    
    return tasks


def prepare_federated_continual_dataloaders(num_clients=3, num_tasks=2, train_batch_size=64, test_batch_size=128, seed=42, num_workers=0, val_ratio=0.1, root="data", task_classes=None):
    """联邦持续学习"""
    # TODO: 实现客户端 + 任务序列加载逻辑
    pass