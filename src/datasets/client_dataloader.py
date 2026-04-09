"""
通用工具模块。

职责：
1. 给一个 dataset 构造 DataLoader
2. 给一组 client datasets 构造多个 DataLoader
3. 给 test dataset 构造 test loader
"""

from torch.utils.data import DataLoader


def make_dataloader(dataset, batch_size=64, shuffle=False, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

def build_train_loader(train_dataset, batch_size=64, num_workers=0):
    """
    构造训练集 DataLoader
    """
    return make_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

def build_client_loaders(client_datasets, batch_size=64, num_workers=0):
    """
    为每个客户端构造训练 DataLoader
    client_datasets:
        {
            0: Subset(...),
            1: Subset(...),
            2: Subset(...)
        }
    """
    client_loaders = {}

    for client_id, dataset in client_datasets.items():
        client_loaders[client_id] = make_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,  # 本质上它就是“给很多个客户端批量创建 train_loader”
            num_workers=num_workers
        )

    return client_loaders


def build_test_loader(test_dataset, batch_size=128, num_workers=0):
    """
    构造全局测试集 DataLoader
    """
    return make_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )