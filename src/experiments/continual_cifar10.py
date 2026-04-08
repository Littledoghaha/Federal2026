"""
持续学习实验（CIFAR-10 版本）

实验场景：
- Task 1: CIFAR-10 前 5 个类别（0-4）
- Task 2: CIFAR-10 后 5 个类别（5-9）

实验目标：
1. 先在 Task 1 上训练模型
2. 再继续学习 Task 2
3. 观察模型在学习新任务后，对旧任务（Task 1）的遗忘情况
4. 在 Phase 2 中加入简单的经验重放（Replay），缓解灾难性遗忘

说明：
- 这里仍然保留“客户端”划分形式，便于和联邦实验结构保持一致
- 但本质上这个实验更接近“持续学习 + 多子数据分块训练”
- 当前经验重放策略比较简单：从 Task 1 训练集里抽取一部分旧样本，
  在 Phase 2 训练时与 Task 2 数据拼接在一起
"""

import os
import json
import random
import torch

from torch.utils.data import Subset, ConcatDataset

from datasets.data_loader import prepare_continual_dataloaders
from datasets.data_split import iid_split_indices, build_client_subsets
from client_dataloader import build_client_loaders
from model import SimpleCNN
from train_eval import train_local, evaluate
from utils.results_standard import save_history_json
from utils.results_continual import save_continual_history_csv, plot_continual_history


def set_seed(seed=42):
    """
    设置随机种子，保证实验尽量可复现。
    """
    random.seed(seed)
    torch.manual_seed(seed)


def run_continual_learning(config, save_dir):
    """
    持续学习主流程（CIFAR-10）

    参数：
    - config: 配置字典
    - save_dir: 结果保存目录

    整体流程：
    1. 准备两个任务的数据
    2. Phase 1：只在 Task 1 上训练
    3. 构造经验重放数据
    4. Phase 2：在 Task 2 上继续训练，同时混入部分 Task 1 数据
    5. 分别评估 Task 1 / Task 2 测试精度
    6. 保存结果
    """
    # =========================
    # 1. 设置随机种子和设备
    # =========================
    set_seed(config["seed"])

    device = torch.device(
        config["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # =========================
    # 2. 准备持续学习任务数据
    # =========================
    # 这里显式指定两个任务的类别划分：
    # task0 -> 类别 0,1,2,3,4
    # task1 -> 类别 5,6,7,8,9
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

    # 为了更直观，把 task0 / task1 重命名为 task1 / task2 变量
    task1_data = task_data["task0"]   # 旧任务：类别 0-4
    task2_data = task_data["task1"]   # 新任务：类别 5-9

    print("\nTask split summary:")
    print(f"Task 1 (classes 0-4): {len(task1_data['train_dataset'])} train samples")
    print(f"Task 2 (classes 5-9): {len(task2_data['train_dataset'])} train samples")

    # =========================
    # 3. 将每个任务的训练集再划分给多个客户端
    # =========================
    # 这样做的目的是让持续学习实验的训练风格，尽量和联邦学习实验结构保持接近
    task1_train_indices = iid_split_indices(
        task1_data["train_dataset"],
        num_clients=config["num_clients"],
        seed=config["seed"]
    )
    task1_client_datasets = build_client_subsets(task1_data["train_dataset"], task1_train_indices)
    task1_client_loaders = build_client_loaders(
        task1_client_datasets,
        batch_size=config["train_batch_size"]
    )

    task2_train_indices = iid_split_indices(
        task2_data["train_dataset"],
        num_clients=config["num_clients"],
        seed=config["seed"]
    )
    task2_client_datasets = build_client_subsets(task2_data["train_dataset"], task2_train_indices)
    task2_client_loaders = build_client_loaders(
        task2_client_datasets,
        batch_size=config["train_batch_size"]
    )

    # 测试集 loader：用于分别测试两个任务
    task1_test_loader = task1_data["test_loader"]
    task2_test_loader = task2_data["test_loader"]

    # =========================
    # 4. 初始化模型
    # =========================
    # 虽然任务按类别分成了两组，但模型仍然输出 10 个类别
    # 因为 CIFAR-10 的标签还是原始的 0~9
    model = SimpleCNN(num_classes=10).to(device)

    # 用于记录实验过程中的指标
    history = {
        "phase": [],           # 当前属于 Task1 阶段还是 Task2 阶段
        "round": [],           # 当前轮次
        "task1_test_acc": [],  # Task 1 测试准确率
        "task2_test_acc": [],  # Task 2 测试准确率
        "forgetting": [],      # 遗忘指标
    }

    # ==========================================================
    # Phase 1: 先学习 Task 1
    # ==========================================================
    print("\n" + "=" * 60)
    print("Phase 1: Training on Task 1 (Classes 0-4)")
    print("=" * 60)

    for round_idx in range(1, config["num_rounds"] + 1):
        print(f"\n--- Round {round_idx}/{config['num_rounds']} ---")

        # 依次在每个“客户端”的 Task 1 数据上训练
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

        # Phase 1 主要看 Task 1 的测试效果
        _, task1_acc = evaluate(model, task1_test_loader, device)
        print(f"Task 1 Test Acc: {task1_acc:.4f}")
        
        history["phase"].append("Task1")
        history["round"].append(round_idx)
        history["task1_test_acc"].append(task1_acc)
        history["task2_test_acc"].append(None)
        history["forgetting"].append(None)

    # Phase 1 结束后记录参考准确率
    task1_reference_acc = max(history["task1_test_acc"])
    print(f"\nTask 1 reference acc for forgetting: {task1_reference_acc:.4f}")
    
    # ==========================================================
    # Phase 1 结束后：构造经验重放数据
    # ==========================================================
    print("\nPreparing replay memory from Task 1...")

    # 经验重放比例：从 Task 1 训练数据中抽取 20% 作为旧任务回放样本
    replay_fraction = 0.2

    # task1_train_indices 是一个字典：
    # {
    #   client0: [索引列表],
    #   client1: [索引列表],
    #   ...
    # }
    # 这里把所有客户端上的 Task 1 训练索引汇总起来
    all_task1_indices = []
    for indices in task1_train_indices.values():
        all_task1_indices.extend(indices)

    replay_size = int(len(all_task1_indices) * replay_fraction)

    # 为了防止 replay_size 意外为 0（例如数据量很小时），做一个最小保护
    replay_size = max(replay_size, 1)

    # 从 Task 1 的训练数据索引中随机抽取一部分
    replay_indices = random.sample(all_task1_indices, replay_size)

    # 用抽样索引从 Task 1 训练集中构造重放数据集
    task1_replay_dataset = Subset(task1_data["train_dataset"], replay_indices)

    print(f"Replay memory size: {len(task1_replay_dataset)} samples")

    # ==========================================================
    # Phase 2: 学习 Task 2，同时加入经验重放
    # ==========================================================
    # 这里的做法是：
    # 对每个客户端，把它自己的 Task 2 数据，与统一的 Task 1 replay 数据拼接起来
    # 然后在这个混合数据集上训练
    task2_client_mixed_datasets = {}

    for client_id in range(config["num_clients"]):
        # 当前客户端的 Task 2 数据
        task2_subset = task2_client_datasets[client_id]

        # 经验重放数据：这里采用“所有客户端共享同一份 replay 数据”的简单做法
        mixed_dataset = ConcatDataset([task2_subset, task1_replay_dataset])
        task2_client_mixed_datasets[client_id] = mixed_dataset

    task2_client_mixed_loaders = build_client_loaders(
        task2_client_mixed_datasets,
        batch_size=config["train_batch_size"]
    )

    print("\n" + "=" * 60)
    print("Phase 2: Continual Learning on Task 2 (Classes 5-9) with Replay")
    print("Observe forgetting on Task 1 while learning Task 2")
    print("=" * 60)

    for round_idx in range(1, config["num_rounds"] + 1):
        print(f"\n--- Round {round_idx}/{config['num_rounds']} ---")

        # 在混合数据（Task 2 + replay）上继续训练
        for client_id, train_loader in task2_client_mixed_loaders.items():
            model, _ = train_local(
                model=model,
                train_loader=train_loader,
                device=device,
                epochs=config["local_epochs"],
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                verbose=False
            )

        # Phase 2 需要同时评估两个任务：
        # - Task 1: 看旧任务是否遗忘
        # - Task 2: 看新任务学得怎么样
        _, task1_in_task2_acc = evaluate(model, task1_test_loader, device)
        _, task2_acc = evaluate(model, task2_test_loader, device)

        print(f"Task 1 Test Acc: {task1_in_task2_acc:.4f} (old task / forgetting indicator)")
        print(f"Task 2 Test Acc: {task2_acc:.4f} (new task)")

        forgetting = task1_reference_acc - task1_in_task2_acc
        history["phase"].append("Task2")
        history["round"].append(round_idx)
        history["task1_test_acc"].append(task1_in_task2_acc)
        history["task2_test_acc"].append(task2_acc)
        history["forgetting"].append(forgetting)
        print(f"Forgetting (Task 1 reference acc - Task 1 in Task 2 acc): {forgetting:.4f}")


    # =========================
    # 5. 保存结果
    # =========================
    save_history_json(history, os.path.join(save_dir, "continual_history.json"))
    save_continual_history_csv(history, os.path.join(save_dir, "continual_history.csv"))
    plot_continual_history(history, save_dir, experiment_name="continual_cifar10")
    torch.save(model.state_dict(), os.path.join(save_dir, "continual_model.pth"))

    # 额外保存一个 summary，便于后续看最终结果
    summary = {
        "experiment_name": f'{config["dataset"]}_continual',
        "num_rounds_per_phase": config["num_rounds"],
        "local_epochs": config["local_epochs"],
        "replay_fraction": replay_fraction,
        "final_task1_test_acc": history["task1_test_acc"][-1],
        "final_task2_test_acc": history["task2_test_acc"][-1],
        "final_forgetting": history["forgetting"][-1],
    }

    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {save_dir}")