'''
负责把一个训练集切成多个 client 的索引
IID 平均切分
'''

from collections import Counter
import numpy as np
from torch.utils.data import Subset

def get_dataset_labels(dataset):
    """
    兼容 CIFAR-10 / SVHN / Subset
    说明：
    - CIFAR-10 一般用 dataset.targets
    - SVHN 一般用 dataset.labels
    - 如果传入的是 Subset，则先递归拿到底层数据集的标签，再按 subset 的索引取出对应部分
    """
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = np.array(dataset.indices)
        labels = get_dataset_labels(base)
        return labels[indices]
    
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    
    if hasattr(dataset, "labels"):
        labels = np.array(dataset.labels)
        labels = np.where(labels == 10, 0, labels)  # SVHN 里数字 0 有时会记成 10，这里统一转成 0
        return labels
    
    raise ValueError("This dataset has no 'targets' or 'labels' attribute.")

def filter_dataset_by_classes(dataset, classes):
    """
    从数据集中筛选指定类别的样本，并返回一个新的 Subset。
    参数：
        dataset: 原始数据集或 Subset
        classes: 需要保留的类别列表，如 [0, 1, 2]
    返回：
        Subset(dataset, indices)
    """
    labels = get_dataset_labels(dataset)
    indices = [i for i, label in enumerate(labels) if label in classes]
    return Subset(dataset, indices)

def iid_split_indices(dataset, num_clients=3, seed=42):
    """
    把 dataset 的样本索引平均随机分给多个客户端
    返回:
        {
            0: [...],
            1: [...],
            2: [...]
        }
    """
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)

    client_indices = {}
    for client_id, split in enumerate(splits):
        client_indices[client_id] = split.tolist()

    return client_indices


def build_client_subsets(dataset, client_indices):
    """
    根据索引字典，生成每个客户端对应的 Subset
    """
    client_datasets = {}
    for client_id, indices in client_indices.items():
        client_datasets[client_id] = Subset(dataset, indices)
    return client_datasets


def get_label_distribution(dataset, indices):
    """
    统计某个客户端里各类别样本数量
    """
    labels = get_dataset_labels(dataset)
    selected_labels = labels[np.array(indices)]

    counter = Counter(selected_labels.tolist())
    return dict(sorted(counter.items()))


def print_client_summary(dataset, client_indices, dataset_name="Dataset"):
    """
    打印每个客户端的数据量和类别分布
    """
    print(f"\n=== {dataset_name} Client Split Summary ===")

    total = 0
    for client_id, indices in client_indices.items():
        total += len(indices)
        label_dist = get_label_distribution(dataset, indices)

        print(f"Client {client_id}: {len(indices)} samples")
        print(f"  Label distribution: {label_dist}")

    print(f"Total samples checked: {total}")