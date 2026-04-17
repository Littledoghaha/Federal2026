"""
数据检查脚本
作用：
1. 检查 CIFAR-10 / SVHN 是否能正常下载和加载
2. 检查 train / val / test 样本数
3. 检查联邦划分是否正常
4. 检查持续学习任务划分是否正常
"""

from datasets.cifar_svhn import load_cifar10, load_svhn
from datasets.data_loader import (
    prepare_centralized_dataloaders,
    prepare_federated_dataloaders,
    prepare_continual_dataloaders,
    prepare_federated_continual_dataloaders,
)
from datasets.data_split import get_dataset_labels


def check_basic_dataset():
    print("\n" + "=" * 60)
    print("1. 检查 CIFAR-10 / SVHN 基本加载")
    print("=" * 60)

    cifar_train, cifar_val, cifar_test = load_cifar10(root="data", val_ratio=0.1, seed=42)
    svhn_train, svhn_val, svhn_test = load_svhn(root="data", val_ratio=0.1, seed=42)

    print(f"CIFAR-10 train size: {len(cifar_train)}")
    print(f"CIFAR-10 val size:   {len(cifar_val)}")
    print(f"CIFAR-10 test size:  {len(cifar_test)}")

    print(f"SVHN train size: {len(svhn_train)}")
    print(f"SVHN val size:   {len(svhn_val)}")
    print(f"SVHN test size:  {len(svhn_test)}")


def check_centralized():
    print("\n" + "=" * 60)
    print("2. 检查集中式 DataLoader")
    print("=" * 60)

    train_loader, val_loader, test_loader = prepare_centralized_dataloaders(
        dataset="svhn",
        batch_size=64,
        test_batch_size=128,
        seed=42,
        num_workers=0,
        val_ratio=0.1
    )

    print(f"train batches: {len(train_loader)}")
    print(f"val batches:   {len(val_loader)}")
    print(f"test batches:  {len(test_loader)}")

    x, y = next(iter(train_loader))
    print(f"train batch x shape: {x.shape}")
    print(f"train batch y shape: {y.shape}")
    print(f"sample labels: {y[:10].tolist()}")


def check_federated():
    print("\n" + "=" * 60)
    print("3. 检查联邦学习数据划分")
    print("=" * 60)

    data_bundle = prepare_federated_dataloaders(
        num_clients=3,
        train_batch_size=64,
        test_batch_size=128,
        seed=42,
        num_workers=0,
        val_ratio=0.1,
        root="data"
    )

    svhn_info = data_bundle["svhn"]

    for cid, loader in svhn_info["client_loaders"].items():
        print(f"SVHN client {cid}: {len(loader.dataset)} samples")

    x, y = next(iter(svhn_info["client_loaders"][0]))
    print(f"client 0 batch x shape: {x.shape}")
    print(f"client 0 batch y shape: {y.shape}")
    print(f"client 0 sample labels: {y[:10].tolist()}")


def check_continual():
    print("\n" + "=" * 60)
    print("4. 检查持续学习任务划分")
    print("=" * 60)

    task_classes = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]

    tasks = prepare_continual_dataloaders(
        dataset="svhn",
        batch_size=64,
        test_batch_size=128,
        seed=42,
        num_workers=0,
        val_ratio=0.1,
        num_tasks=3,
        task_classes=task_classes
    )

    for task_name, task_info in tasks.items():
        labels = get_dataset_labels(task_info["train_dataset"])
        unique_labels = sorted(set(labels.tolist()))
        print(f"{task_name}:")
        print(f"  train size: {len(task_info['train_dataset'])}")
        print(f"  val size:   {len(task_info['val_dataset'])}")
        print(f"  test size:  {len(task_info['test_dataset'])}")
        print(f"  classes:    {unique_labels}")


def check_federated_continual():
    print("\n" + "=" * 60)
    print("5. 检查联邦持续学习任务划分")
    print("=" * 60)

    task_classes = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]

    tasks = prepare_federated_continual_dataloaders(
        dataset="svhn",
        num_clients=3,
        num_tasks=3,
        train_batch_size=64,
        test_batch_size=128,
        seed=42,
        num_workers=0,
        val_ratio=0.1,
        root="data",
        task_classes=task_classes
    )

    for task_name, task_info in tasks.items():
        labels = get_dataset_labels(task_info["train_dataset"])
        unique_labels = sorted(set(labels.tolist()))
        print(f"{task_name}:")
        print(f"  train size: {len(task_info['train_dataset'])}")
        print(f"  classes:    {unique_labels}")
        for cid, ds in task_info["client_datasets"].items():
            print(f"    client {cid}: {len(ds)} samples")


if __name__ == "__main__":
    check_basic_dataset()
    check_centralized()
    check_federated()
    check_continual()
    check_federated_continual()