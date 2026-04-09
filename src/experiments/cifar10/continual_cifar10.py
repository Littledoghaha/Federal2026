"""
CIFAR-10 版本持续学习调用脚本

负责：
- 载入数据集、划分任务和客户端
- 初始化模型
- 调用 core.continual.run_continual_learning

保证函数名、打印内容和行为与原先代码一致。
"""

import torch
import os
import random
from datasets.data_loader import prepare_continual_dataloaders
from datasets.data_split import iid_split_indices, build_client_subsets
from core.continual import run_continual_learning
from core.model import SimpleCNN


def run_continual(config, save_dir):
    """
    CIFAR-10 持续学习调用函数。
    """

    def set_seed(seed=42):
        random.seed(seed)
        torch.manual_seed(seed)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from types import SimpleNamespace
    model_args = SimpleNamespace(
        lambda_l1=config.get("lambda_l1", 1e-4),
        lambda_mask=config.get("lambda_mask", 1e-4),
        device=device,
    )

    # 准备两个任务的数据集，task0是类别0-4，task1是类别5-9
    task_data = prepare_continual_dataloaders(
        dataset="cifar10",
        batch_size=config["train_batch_size"],
        test_batch_size=config["test_batch_size"],
        seed=config["seed"],
        num_workers=0,
        val_ratio=0.1,
        num_tasks=2,
        task_classes=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    )

    task1_data = task_data["task0"]  # 旧任务 Task 1（类别 0-4）
    task2_data = task_data["task1"]  # 新任务 Task 2（类别 5-9）

    print("\nTask split summary:")
    # 打印训练样本数量，方便核实
    print(f"Task 1 (classes 0-4): {len(task1_data['train_dataset'])} train samples")
    print(f"Task 2 (classes 5-9): {len(task2_data['train_dataset'])} train samples")

    # 将每个任务训练集进一步划分给多个客户端，保持和联邦学习类似的结构
    task1_train_indices = iid_split_indices(
        task1_data["train_dataset"], num_clients=config["num_clients"], seed=config["seed"]
    )
    task1_client_datasets = build_client_subsets(task1_data["train_dataset"], task1_train_indices)

    task2_train_indices = iid_split_indices(
        task2_data["train_dataset"], num_clients=config["num_clients"], seed=config["seed"]
    )
    task2_client_datasets = build_client_subsets(task2_data["train_dataset"], task2_train_indices)

    # 测试集加载器
    task1_test_loader = task1_data["test_loader"]
    task2_test_loader = task2_data["test_loader"]

    # 初始化模型
    # 注意模型输出类别数仍是10，因为标签未变，只是训练数据分成了两个阶段
    model = SimpleCNN(args=model_args, num_classes=10, in_channels=3)

    # 调用核心算法
    history, summary = run_continual_learning(
        model=model,
        task1_client_datasets=task1_client_datasets,
        task1_train_indices=task1_train_indices,
        task1_test_loader=task1_test_loader,
        task2_client_datasets=task2_client_datasets,
        task2_test_loader=task2_test_loader,
        config=config,
        save_dir=save_dir,
    )
    return history, summary