'''
负责：

把一个训练集切成多个 client 的索引
你现在大概率是 IID 平均切分
'''


from collections import Counter
import numpy as np
from torch.utils.data import Subset


def get_dataset_labels(dataset):
    """
    兼容 CIFAR-10 和 SVHN
    CIFAR-10: dataset.targets
    SVHN: dataset.labels
    """
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)

    if hasattr(dataset, "labels"):
        labels = np.array(dataset.labels)

        # SVHN 里数字 0 有时会记成 10，这里统一转成 0
        labels = np.where(labels == 10, 0, labels)
        return labels

    raise ValueError("This dataset has no 'targets' or 'labels' attribute.")


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