"""
CIFAR-10 版本联邦持续学习调用脚本

负责：
- 载入联邦持续学习所需的数据
- 初始化全局模型
- 调用 core.fed_continual.run_federated_continual_learning

本版本已改为：
- 支持任意 task 数
- task 划分由 config["task_classes"] 控制
"""

import torch
import random
import numpy as np
from types import SimpleNamespace

from datasets.data_loader import prepare_federated_continual_dataloaders
from core.model import SimpleCNN
from core.fed_continual import run_federated_continual_learning


def run_federated_continual(config, save_dir):
    """
    CIFAR-10 联邦持续学习调用函数。
    """

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_args = SimpleNamespace(
        lambda_l1=config.get("lambda_l1", 1e-4),
        lambda_mask=config.get("lambda_mask", 1e-4),
        device=device,
    )

    # =========================
    # 从配置中读取任务划分
    # =========================
    task_classes = config.get(
        "task_classes",
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    )
    num_tasks = len(task_classes)

    # =========================
    # 准备联邦持续学习任务数据
    # 每个 task 都会包含：
    # - train_dataset
    # - client_indices
    # - client_datasets
    # - client_loaders
    # - val_loader
    # - test_loader
    # =========================
    task_data = prepare_federated_continual_dataloaders(
        dataset="cifar10",
        num_clients=config["num_clients"],
        num_tasks=num_tasks,
        train_batch_size=config["train_batch_size"],
        test_batch_size=config["test_batch_size"],
        seed=config["seed"],
        num_workers=0,
        val_ratio=0.1,
        task_classes=task_classes,
    )

    print("\nTask split summary:")

    # 用统一列表保存所有任务
    all_tasks = []

    for task_id in range(num_tasks):
        task_key = f"task{task_id}"
        one_task = task_data[task_key]

        print(
            f"Task {task_id + 1} (classes {task_classes[task_id]}): "
            f"{len(one_task['train_dataset'])} train samples"
        )

        # 打印每个任务在各客户端上的样本数
        for cid, ds in one_task["client_datasets"].items():
            print(f"  client {cid}: {len(ds)} samples")

        all_tasks.append({
            "task_id": task_id,
            "classes": task_classes[task_id],
            "train_dataset": one_task["train_dataset"],
            "val_loader": one_task["val_loader"],
            "test_loader": one_task["test_loader"],
            "client_indices": one_task["client_indices"],
            "client_datasets": one_task["client_datasets"],
            "client_loaders": one_task["client_loaders"],
        })

    # 初始化全局模型
    model = SimpleCNN(args=model_args, num_classes=10, in_channels=3).to(device)

    # 调用联邦持续学习核心算法
    history, summary = run_federated_continual_learning(
        global_model=model,
        tasks=all_tasks,
        config=config,
        save_dir=save_dir,
    )

    return history, summary