"""
持续学习实验（CIFAR-10 版本）

场景：模拟持续学习中的"任务序列"
- Task 1: CIFAR-10 前 5 个类别（0-4）
- Task 2: CIFAR-10 后 5 个类别（5-9）

观察指标：
- Task 1 准确率在学习 Task 2 后的下降（遗忘）
- Task 2 准确率
"""

import torch
import os
import numpy as np
from torch.utils.data import Subset
from datasets.data_split import iid_split_indices, build_client_subsets
from client_dataloader import build_client_loaders, build_test_loader
from model import SimpleCNN
from train_eval import train_local, evaluate
from utils.results_utils import save_history_json, save_history_csv
import json

def split_cifar10_by_class(train_set, val_set, test_set, task1_classes=[0,1,2,3,4]):
    """
    将 CIFAR-10 按类别分成两个任务
    
    Task 1: classes [0,1,2,3,4]
    Task 2: classes [5,6,7,8,9]
    """
    task2_classes = [i for i in range(10) if i not in task1_classes]
    
    def filter_by_classes(dataset, classes):
        """筛选出指定类别的样本"""
        indices = []
        labels = np.array(dataset.dataset.targets) if isinstance(dataset, Subset) else np.array(dataset.targets)
        
        for idx, label in enumerate(labels):
            if label in classes:
                indices.append(idx)
        return Subset(dataset, indices)
    
    # 分割训练集
    task1_train = filter_by_classes(train_set, task1_classes)
    task2_train = filter_by_classes(train_set, task2_classes)
    
    # 分割验证集
    task1_val = filter_by_classes(val_set, task1_classes)
    task2_val = filter_by_classes(val_set, task2_classes)
    
    # 分割测试集
    task1_test = filter_by_classes(test_set, task1_classes)
    task2_test = filter_by_classes(test_set, task2_classes)
    
    return {
        "task1": {"train": task1_train, "val": task1_val, "test": task1_test},
        "task2": {"train": task2_train, "val": task2_val, "test": task2_test},
    }

def run_continual_learning(config, save_dir):
    """
    持续学习主流程（CIFAR-10 单数据集）
    """
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ========== 数据准备 ==========
    # 加载 CIFAR-10（包含验证集）
    from datasets.data_loader import load_cifar10_with_val

    train_set, val_set, test_set = load_cifar10_with_val(root="data", val_ratio=0.1, seed=config["seed"])
    
    # 按类别分成两个任务
    task_data = split_cifar10_by_class(train_set, val_set, test_set)
    
    # Task 1 数据
    task1_train_indices = iid_split_indices(task_data["task1"]["train"], num_clients=config["num_clients"], seed=config["seed"])
    task1_client_datasets = build_client_subsets(task_data["task1"]["train"], task1_train_indices)
    task1_client_loaders = build_client_loaders(task1_client_datasets, batch_size=config["train_batch_size"])
    task1_val_loader = build_test_loader(task_data["task1"]["val"], batch_size=config["test_batch_size"])
    task1_test_loader = build_test_loader(task_data["task1"]["test"], batch_size=config["test_batch_size"])
    
    # Task 2 数据
    task2_train_indices = iid_split_indices(task_data["task2"]["train"], num_clients=config["num_clients"], seed=config["seed"])
    task2_client_datasets = build_client_subsets(task_data["task2"]["train"], task2_train_indices)
    task2_client_loaders = build_client_loaders(task2_client_datasets, batch_size=config["train_batch_size"])
    task2_val_loader = build_test_loader(task_data["task2"]["val"], batch_size=config["test_batch_size"])
    task2_test_loader = build_test_loader(task_data["task2"]["test"], batch_size=config["test_batch_size"])
    
    print(f"\nTask 1 (Classes 0-4): {len(task_data['task1']['train'])} train samples")
    print(f"Task 2 (Classes 5-9): {len(task_data['task2']['train'])} train samples")
    
    # ========== 模型初始化 ==========
    model = SimpleCNN(num_classes=10).to(device)
    
    history = {
        "phase": [],
        "round": [],
        "task1_test_acc": [],
        "task2_test_acc": [],
    }
    
    # ========== Phase 1: 在 Task 1 上训练 ==========
    print("\n" + "="*60)
    print("Phase 1: Training on Task 1 (Classes 0-4)")
    print("="*60)
    
    for round_idx in range(1, config["num_rounds"] + 1):
        print(f"\n--- Round {round_idx}/{config['num_rounds']} ---")
        
        # 客户端本地训练
        for client_id, train_loader in task1_client_loaders.items():
            model, _ = train_local(
                model=model,
                train_loader=train_loader,
                device=device,
                epochs=config["local_epochs"],
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                verbose=False
            )
        
        # 评估 Task 1 性能
        _, task1_acc = evaluate(model, task1_test_loader, device)
        print(f"Task 1 Test Acc: {task1_acc:.4f}")
        
        history["phase"].append("Task1")
        history["round"].append(round_idx)
        history["task1_test_acc"].append(task1_acc)
        history["task2_test_acc"].append(None)
    
    # ========== Phase 2: 在 Task 2 上继续学习 ==========
    print("\n" + "="*60)
    print("Phase 2: Continual Learning on Task 2 (Classes 5-9)")
    print("观察模型对 Task 1 的遗忘程度")
    print("="*60)
    
    for round_idx in range(1, config["num_rounds"] + 1):
        print(f"\n--- Round {round_idx}/{config['num_rounds']} ---")
        
        # 客户端在 Task 2 上继续训练
        for client_id, train_loader in task2_client_loaders.items():
            model, _ = train_local(
                model=model,
                train_loader=train_loader,
                device=device,
                epochs=config["local_epochs"],
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                verbose=False
            )
        
        # 同时评估两个任务
        _, task1_acc = evaluate(model, task1_test_loader, device)
        _, task2_acc = evaluate(model, task2_test_loader, device)
        
        print(f"Task 1 Test Acc: {task1_acc:.4f} (旧任务 - 遗忘指标)")
        print(f"Task 2 Test Acc: {task2_acc:.4f} (新任务)")
        
        history["phase"].append("Task2")
        history["round"].append(round_idx)
        history["task1_test_acc"].append(task1_acc)
        history["task2_test_acc"].append(task2_acc)
    
    # ========== 保存结果 ==========
    save_history_json(history, os.path.join(save_dir, "continual_history.json"))
    save_history_csv(history, os.path.join(save_dir, "continual_history.csv"))
    torch.save(model.state_dict(), os.path.join(save_dir, "continual_model.pth"))
    
    print(f"\n✓ Results saved to: {save_dir}")