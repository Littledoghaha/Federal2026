"""
持续学习核心算法模块

本版本已从“固定 2 个任务”改为“支持任意 task 数”的通用版本。

主要思想：
1. 按任务顺序依次训练：Task1 -> Task2 -> Task3 -> ...
2. 第一个任务正常训练
3. 后续任务若开启 replay，则从所有旧任务中抽取一部分样本进行经验重放
4. 每轮训练后评估当前已见过的所有任务
5. 遗忘指标采用“旧任务平均遗忘”：
   forgetting = mean(reference_acc_i - current_acc_i), i 属于所有旧任务
"""

import os
import json
import random
import torch
from torch.utils.data import Subset, ConcatDataset

from core.train_eval import train_local, evaluate
from datasets.client_dataloader import build_client_loaders
from utils.results_standard import save_history_json
from utils.results_continual import save_continual_history_csv, plot_continual_history


def run_continual_learning(model, tasks, config, save_dir):
    """
    持续学习主流程（核心算法，通用 task 数版本）

    参数说明：
    - model: 已初始化的模型实例，对所有客户端共享
    - tasks: 所有任务的信息列表，每个元素格式为：
        {
            "task_id": ...,
            "classes": ...,
            "train_dataset": ...,
            "val_loader": ...,
            "test_loader": ...,
            "train_indices": ...,
            "client_datasets": ...
        }
    - config: 实验配置，包含超参数等
    - save_dir: 结果保存目录

    实验流程：
    1. 按任务顺序依次训练
    2. 第一个任务单独训练
    3. 从第二个任务开始，可加入 replay
    4. 每个 round 结束后评估所有已学习任务
    5. 保存训练历史和模型
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
    # 经验重放样本比例，默认从旧任务训练集中抽取 20%
    replay_fraction = config.get("replay_fraction", 0.2)

    num_tasks = len(tasks)

    # =========================
    # 用于记录训练过程指标
    # 注意：这里改为动态生成 task1/task2/task3/... 的指标字段
    # =========================
    history = {
        "phase": [],
        "round": [],
        "forgetting": [],
    }

    for task_idx in range(num_tasks):
        history[f"task{task_idx + 1}_val_acc"] = []
        history[f"task{task_idx + 1}_test_acc"] = []

    # 记录每个任务在其训练结束时的参考精度，用于后续计算遗忘
    task_reference_acc = {}

    def append_history(task_phase_name, round_idx, eval_results, forgetting_value):
        """
        将当前 round 的评估结果写入 history。

        参数：
        - task_phase_name: 当前属于哪个任务阶段，例如 "Task1"
        - round_idx: 当前阶段内的轮次
        - eval_results: 当前已见任务的评估结果
        - forgetting_value: 当前遗忘指标
        """
        history["phase"].append(task_phase_name)
        history["round"].append(round_idx)
        history["forgetting"].append(forgetting_value)

        # 对所有任务统一补齐记录：
        # 已评估到的任务写真实值，尚未出现的任务写 None
        for eval_task_idx in range(num_tasks):
            val_key = f"task{eval_task_idx + 1}_val_acc"
            test_key = f"task{eval_task_idx + 1}_test_acc"

            if eval_task_idx in eval_results:
                history[val_key].append(eval_results[eval_task_idx]["val_acc"])
                history[test_key].append(eval_results[eval_task_idx]["test_acc"])
            else:
                history[val_key].append(None)
                history[test_key].append(None)

    def evaluate_seen_tasks(model, tasks, seen_task_ids, device):
        """
        评估当前模型在所有“已见任务”上的 val/test 表现。
        返回：
            {
                0: {"val_acc": ..., "test_acc": ...},
                1: {"val_acc": ..., "test_acc": ...},
                ...
            }
        """
        results = {}
        for tid in seen_task_ids:
            _, val_acc = evaluate(model, tasks[tid]["val_loader"], device)
            _, test_acc = evaluate(model, tasks[tid]["test_loader"], device)
            results[tid] = {
                "val_acc": val_acc,
                "test_acc": test_acc,
            }
        return results

    def build_replay_dataset_from_previous_tasks(previous_tasks, replay_fraction, seed):
        """
        从所有旧任务中抽样，构造 replay 数据集。

        做法：
        - 对每个旧任务，按照 replay_fraction 抽一部分训练样本
        - 再将这些旧任务 replay subset 合并起来

        返回：
        - 如果只有一个旧任务，直接返回该 Subset
        - 如果有多个旧任务，返回 ConcatDataset
        """
        replay_subsets = []
        random.seed(seed)

        for old_task in previous_tasks:
            all_indices = []
            for indices in old_task["train_indices"].values():
                all_indices.extend(indices)

            replay_size = max(1, int(len(all_indices) * replay_fraction))
            replay_indices = random.sample(all_indices, replay_size)

            replay_subset = Subset(old_task["train_dataset"], replay_indices)
            replay_subsets.append(replay_subset)

        if len(replay_subsets) == 1:
            return replay_subsets[0]
        return ConcatDataset(replay_subsets)

    # ==========================================================
    # 按任务顺序训练：Task1 -> Task2 -> Task3 -> ...
    # ==========================================================
    for task_idx, current_task in enumerate(tasks):
        print("\n" + "=" * 60)
        print(f"Phase {task_idx + 1}: Training on Task {task_idx + 1} {current_task['classes']}")
        print("=" * 60)

        current_client_datasets = current_task["client_datasets"]

        # =========================
        # 如果是第一个任务，直接使用当前任务数据训练
        # 如果不是第一个任务，可选加入 replay
        # =========================
        if task_idx == 0:
            mixed_client_datasets = current_client_datasets
        else:
            if use_replay:
                print("\n经验重放开启")

                previous_tasks = tasks[:task_idx]
                replay_dataset = build_replay_dataset_from_previous_tasks(
                    previous_tasks=previous_tasks,
                    replay_fraction=replay_fraction,
                    seed=config["seed"]
                )
                print(f"Replay memory size: {len(replay_dataset)} samples")

                # 将当前任务数据与 replay 数据拼接
                # 这里采用“所有客户端共享同一份 replay 数据”的简单策略
                mixed_client_datasets = {}
                for client_id, current_subset in current_client_datasets.items():
                    mixed_dataset = ConcatDataset([current_subset, replay_dataset])
                    mixed_client_datasets[client_id] = mixed_dataset
            else:
                print("\n经验重放关闭，仅使用当前任务数据训练")
                mixed_client_datasets = current_client_datasets

        # 为当前阶段的客户端构造 DataLoader
        current_client_loaders = build_client_loaders(
            mixed_client_datasets,
            batch_size=config["train_batch_size"]
        )

        # =========================
        # 当前任务阶段的训练轮次
        # =========================
        for round_idx in range(1, config["num_rounds"] + 1):
            print(f"\n--- Round {round_idx}/{config['num_rounds']} ---")

            # 依次在每个客户端的当前任务数据上训练
            # 注意：这里仍然是“顺序客户端训练”风格
            # 它不是联邦聚合，而是持续学习实验中为了保持结构一致采用的客户端划分形式
            for client_id, train_loader in current_client_loaders.items():
                model, _ = train_local(
                    model=model,
                    train_loader=train_loader,
                    device=device,
                    epochs=config["local_epochs"],
                    lr=config["lr"],
                    weight_decay=config["weight_decay"],
                    verbose=False,
                )

            # 评估当前已见过的所有任务
            seen_task_ids = list(range(task_idx + 1))
            eval_results = evaluate_seen_tasks(model, tasks, seen_task_ids, device)

            for tid in seen_task_ids:
                print(
                    f"Task {tid + 1} Val  Acc: {eval_results[tid]['val_acc']:.4f} | "
                    f"Test Acc: {eval_results[tid]['test_acc']:.4f}"
                )

            # =========================
            # 计算遗忘指标
            # 第一个任务阶段没有遗忘概念
            # 从第二个任务开始，对所有旧任务求平均遗忘
            # =========================
            if task_idx == 0:
                forgetting = None
            else:
                forgetting_list = []
                for old_tid in range(task_idx):
                    ref_acc = task_reference_acc[old_tid]
                    cur_acc = eval_results[old_tid]["test_acc"]
                    forgetting_list.append(ref_acc - cur_acc)

                forgetting = sum(forgetting_list) / len(forgetting_list)
                print(f"Average Forgetting: {forgetting:.4f}")

            # 保存本轮结果
            append_history(
                task_phase_name=f"Task{task_idx + 1}",
                round_idx=round_idx,
                eval_results=eval_results,
                forgetting_value=forgetting
            )

        # =========================
        # 当前任务训练结束后，记录该任务参考精度
        # 这里采用“该任务训练结束时的 test acc”作为 reference
        # 后续阶段据此计算该任务被遗忘了多少
        # =========================
        final_eval = evaluate_seen_tasks(model, tasks, [task_idx], device)
        task_reference_acc[task_idx] = final_eval[task_idx]["test_acc"]
        print(f"\nTask {task_idx + 1} reference acc for forgetting: {task_reference_acc[task_idx]:.4f}")

    # =========================
    # 保存结果
    # =========================
    save_history_json(history, os.path.join(save_dir, "continual_history.json"))
    save_continual_history_csv(history, os.path.join(save_dir, "continual_history.csv"))

    experiment_name = "continual_replay" if use_replay else "continual_no_replay"
    plot_continual_history(history, save_dir, experiment_name=experiment_name)

    torch.save(model.state_dict(), os.path.join(save_dir, "continual_model.pth"))

    # 额外保存一个 summary，便于后续看最终结果
    summary = {
        "experiment_name": f'{config["dataset"]}_continual',
        "use_replay": use_replay,
        "num_tasks": num_tasks,
        "num_rounds_per_phase": config["num_rounds"],
        "local_epochs": config["local_epochs"],
        "replay_fraction": replay_fraction if use_replay else 0.0,
        "task_reference_acc": task_reference_acc,
        "final_forgetting": history["forgetting"][-1],
    }

    # 动态记录每个任务的最终测试精度
    for task_idx in range(num_tasks):
        summary[f"final_task{task_idx + 1}_test_acc"] = history[f"task{task_idx + 1}_test_acc"][-1]

    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {save_dir}")
    return history, summary