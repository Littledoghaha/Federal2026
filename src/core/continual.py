import os
import json
import random
import copy
import torch
from torch.utils.data import Subset, ConcatDataset

from core.train_eval import train_local, evaluate
from utils.results_standard import save_history_json
from utils.results_continual import save_continual_history_csv, plot_continual_history


def run_continual_learning(
    model,
    task1_client_datasets,
    task1_train_indices,
    task1_test_loader,
    task2_client_datasets,
    task2_test_loader,
    config,
    save_dir
):
    """
    持续学习主流程（核心算法）

    参数说明：
    - model: 已初始化的模型实例，对所有客户端共享
    - task1_client_datasets: Task 1 训练数据按客户端划分的字典或列表
    - task1_train_indices: Task 1 每客户端的训练样本索引，方便经验重放采样
    - task1_test_loader: Task 1 测试集 DataLoader
    - task2_client_datasets: Task 2 训练数据按客户端划分的字典或列表
    - task2_test_loader: Task 2 测试集 DataLoader
    - config: 实验配置，包含超参数等
    - save_dir: 结果保存目录

    实验流程：
    1. Phase 1：训练 Task 1（旧任务）
    2. 构造经验重放数据（如开启），从 Task 1 数据中抽取部分样本
    3. Phase 2：训练 Task 2（新任务），混合经验重放数据缓解灾难性遗忘
    4. 训练过程中评估 Task 1 和 Task 2 准确率，计算遗忘指标
    5. 保存训练历史和模型

    本函数与具体数据集无关，便于多数据集兼容使用。
    """

    def set_seed(seed=42):
        """
        设置随机种子，保证实验尽量可复现。
        """
        random.seed(seed)
        torch.manual_seed(seed)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 是否启用经验重放策略，控制新任务学习时加入旧任务样本
    use_replay = config.get("use_replay", True)
    # 经验重放样本比例，默认从 Task 1 训练集中抽取20%
    replay_fraction = config.get("replay_fraction", 0.2)

    # 构造每个客户端的DataLoader，统一batch size
    from datasets.client_dataloader import build_client_loaders

    # Task 1 每客户端训练loader
    task1_client_loaders = build_client_loaders(
        task1_client_datasets, batch_size=config["train_batch_size"]
    )

    # 经验重放数据暂时为空
    task2_client_mixed_datasets = None

    print("\n" + "=" * 60)
    print("Phase 1: Training on Task 1 (Classes of old task)")
    print("=" * 60)

    # 记录训练过程指标
    history = {
        "phase": [],
        "round": [],
        "task1_test_acc": [],
        "task2_test_acc": [],
        "forgetting": [],
    }

    # =========================
    # Phase 1: 先训练旧任务 Task 1
    # =========================
    for round_idx in range(1, config["num_rounds"] + 1):
        print(f"\n--- Round {round_idx}/{config['num_rounds']} ---")

        # 依次在每个客户端的 Task 1 数据上训练
        for client_id, train_loader in task1_client_loaders.items():
            model, _ = train_local(
                model=model,
                train_loader=train_loader,
                device=device,
                epochs=config["local_epochs"],
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                verbose=False,
            )
        # 评估 Task 1 的测试准确率
        _, task1_acc = evaluate(model, task1_test_loader, device)
        print(f"Task 1 Test Acc: {task1_acc:.4f}")

        # 记录指标
        history["phase"].append("Task1")
        history["round"].append(round_idx)
        history["task1_test_acc"].append(task1_acc)
        history["task2_test_acc"].append(None)  # Task2尚未训练
        history["forgetting"].append(None)  # 遗忘此时无意义

    # Phase 1 结束后记录 Task 1 最高准确率作为参考，计算遗忘用
    task1_reference_acc = max(history["task1_test_acc"])
    print(f"\nTask 1 reference acc for forgetting: {task1_reference_acc:.4f}")

    # =========================
    # Phase 2 开始前，构造经验重放数据
    # =========================
    if use_replay:
        print(f"\n经验重放开启")

        # 汇总所有客户端 Task 1 的训练索引，方便全局采样
        all_task1_indices = []
        for indices in task1_train_indices.values():
            all_task1_indices.extend(indices)

        replay_size = max(int(len(all_task1_indices) * replay_fraction), 1)  # 避免为0
        # 从 Task 1 训练索引中随机采样
        replay_indices = random.sample(all_task1_indices, replay_size)

        # 这里需要从客户端集合中找到底层完整的 Task 1 训练集，构造重放Subset
        all_task1_dataset = None
        for ds in task1_client_datasets.values():
            if isinstance(ds, torch.utils.data.Subset):
                all_task1_dataset = ds.dataset
            else:
                all_task1_dataset = ds
            break

        task1_replay_dataset = Subset(all_task1_dataset, replay_indices)

        print(f"Replay memory size: {len(task1_replay_dataset)} samples")

        # =========================
        # Phase 2：构造混合数据集，Task 2 数据 + Replay 旧任务数据
        # 这里简单让所有客户端共享同一份 replay 数据
        # =========================
        task2_client_mixed_datasets = {}
        for client_id, task2_subset in task2_client_datasets.items():
            mixed_dataset = ConcatDataset([task2_subset, task1_replay_dataset])
            task2_client_mixed_datasets[client_id] = mixed_dataset
    else:
        print(f"\n经验重放关闭，Phase 2 只用 Task 2 数据训练")
        task2_client_mixed_datasets = task2_client_datasets

    # 构造 Phase 2 各客户端的混合数据加载器
    task2_client_loaders = build_client_loaders(
        task2_client_mixed_datasets, batch_size=config["train_batch_size"]
    )

    print("\n" + "=" * 60)
    print("Phase 2: Continual Learning on Task 2 with Replay")
    print("Observe forgetting on Task 1 while learning Task 2")
    print("=" * 60)

    # 评估 Phase 2 训练开始前 Task 1 精度
    _, task1_before = evaluate(model, task1_test_loader, device)
    print(f"\nTask1 acc before Phase2 training: {task1_before:.4f}")

    # =========================
    # Phase 2: 训练新任务 Task 2，监控 Task 1 遗忘情况
    # =========================
    for round_idx in range(1, config["num_rounds"] + 1):
        print(f"\n--- Round {round_idx}/{config['num_rounds']} ---")

        for client_id, train_loader in task2_client_loaders.items():
            print(f"\n--- Training on client {client_id} ---")
            model, _ = train_local(
                model=model,
                train_loader=train_loader,
                device=device,
                epochs=config["local_epochs"],
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                verbose=False,
            )

            # 调试：打印 Task1 测试集预测前20个结果及标签
            model.eval()
            with torch.no_grad():
                for images, labels in task1_test_loader:
                    outputs = model(images.to(device))
                    preds = outputs.argmax(dim=1)
                    print("Predictions (first 20):", preds[:20].tolist())
                    print("Labels (first 20):     ", labels[:20].tolist())
                    break

            # 训练完当前客户端后，立刻评估 Task1 准确率
            _, acc_after_client = evaluate(model, task1_test_loader, device)
            print(f"Task1 acc after client {client_id}: {acc_after_client:.4f}")

            # 打印最后一层全连接层权重均值（分任务类别），辅助检查参数变化
            last_linear = model.fc2
            weights = last_linear.weight.data
            print("Weights mean (classes 0-4):", weights[:5].mean().item())
            print("Weights mean (classes 5-9):", weights[5:].mean().item())

        # 每个round结束时同时评估两个任务准确率
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

    # 保存训练记录和模型
    save_history_json(history, os.path.join(save_dir, "continual_history.json"))
    save_continual_history_csv(history, os.path.join(save_dir, "continual_history.csv"))
    experiment_name = "continual_replay" if use_replay else "continual_no_replay"
    plot_continual_history(history, save_dir, experiment_name=experiment_name)
    torch.save(model.state_dict(), os.path.join(save_dir, "continual_model.pth"))

    # 保存 summary，方便后续分析
    summary = {
        "experiment_name": f'{config["dataset"]}_continual',
        "use_replay": use_replay,
        "num_rounds_per_phase": config["num_rounds"],
        "local_epochs": config["local_epochs"],
        "replay_fraction": replay_fraction if use_replay else 0.0,
        "task1_reference_acc": task1_reference_acc,
        "final_task1_test_acc": history["task1_test_acc"][-1],
        "final_task2_test_acc": history["task2_test_acc"][-1],
        "final_forgetting": history["forgetting"][-1],
    }

    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {save_dir}")
    return history, summary