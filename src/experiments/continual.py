"""
持续学习实验

负责：
- 载入数据集、划分任务和客户端
- 初始化模型
- 调用 core.continual.run_continual_learning

- 支持任意 task 数，task 划分由 config["task_classes"] 控制
"""

import torch
import random
import numpy as np
from types import SimpleNamespace

from datasets.data_loader import prepare_continual_dataloaders
from datasets.data_split import iid_split_indices, build_client_subsets
from core.continual import run_continual_learning
from core.model import SimpleCNN


def run_continual(config, save_dir):
    """
    持续学习调用函数。

    说明：
    - 这里只负责“数据准备 + 模型初始化 + 调用核心算法”
    - 真正的持续学习训练流程放在 core/continual.py 中
    """

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 模型中分解参数层所需的超参数
    model_args = SimpleNamespace(
        lambda_l1=config.get("lambda_l1", 1e-4),
        lambda_mask=config.get("lambda_mask", 1e-4),
        device=device
    )

    # =========================
    # 从配置中读取任务划分
    # 如果 config 中没有 task_classes，就默认使用 2 task 划分
    # =========================
    task_classes = config.get(
        "task_classes",
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    )
    num_tasks = len(task_classes)

    # =========================
    # 准备持续学习任务数据
    # 每个 task 都会包含：
    # - train_dataset
    # - val_loader
    # - test_loader
    # =========================
    task_data = prepare_continual_dataloaders(
        dataset=config["dataset"],
        batch_size=config["train_batch_size"],
        test_batch_size=config["test_batch_size"],
        seed=config["seed"],
        num_workers=0,
        val_ratio=0.1,
        num_tasks=num_tasks,
        task_classes=task_classes,
    )

    print("\nTask split summary:")

    # 用一个列表统一保存所有任务，后续直接传给 core 层
    all_tasks = []

    # =========================
    # 遍历每一个任务：
    # 1. 打印任务样本数
    # 2. 把任务训练集再划分给多个客户端
    # 3. 组织成统一结构，交给 run_continual_learning
    # =========================
    for task_id in range(num_tasks):
        task_key = f"task{task_id}"
        one_task = task_data[task_key]

        print(
            f"Task {task_id + 1} (classes {task_classes[task_id]}): "
            f"{len(one_task['train_dataset'])} train samples"
        )

        # 将当前任务训练集进一步划分给多个客户端
        # 保持和联邦学习类似的结构，便于后续实验对比
        train_indices = iid_split_indices(
            one_task["train_dataset"],
            num_clients=config["num_clients"],
            seed=config["seed"]
        )
        client_datasets = build_client_subsets(
            one_task["train_dataset"],
            train_indices
        )

        all_tasks.append({
            "task_id": task_id,
            "classes": task_classes[task_id],
            "train_dataset": one_task["train_dataset"],
            "val_loader": one_task["val_loader"],
            "test_loader": one_task["test_loader"],
            "train_indices": train_indices,
            "client_datasets": client_datasets,
        })

    # 初始化模型
    # 注意模型输出类别数仍是10，因为标签未变，只是训练数据分成多个阶段
    model = SimpleCNN(args=model_args, num_classes=10, in_channels=3)

    # 调用核心算法
    history, summary = run_continual_learning(
        model=model,
        tasks=all_tasks,
        config=config,
        save_dir=save_dir,
    )

    return history, summary