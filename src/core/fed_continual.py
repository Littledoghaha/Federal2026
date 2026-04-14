"""
联邦持续学习核心算法模块

本版本已从“固定 2 个任务”改为“支持任意 task 数”的通用版本。

主要思想：
1. 按任务顺序依次训练：Task1 -> Task2 -> Task3 -> ...
2. 每个任务阶段内部执行标准 FedAvg：
   - 服务器下发全局模型
   - 客户端本地训练
   - 服务器聚合
3. 第一个任务正常训练
4. 后续任务若开启 replay，则从所有旧任务中抽取一部分样本进行经验重放
5. 每轮聚合后评估当前已见过的所有任务
6. 遗忘指标采用“旧任务平均遗忘”
"""

import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Subset, ConcatDataset
from core.fedavg import fedavg_round
from core.train_eval import evaluate
from datasets.client_dataloader import build_client_loaders
from utils.results_standard import save_history_json
from utils.results_continual import save_continual_history_csv, plot_continual_history


def run_federated_continual_learning(global_model, tasks, config, save_dir):
    """
    联邦持续学习主流程（核心算法，通用 task 数版本）

    参数说明：
    - global_model: 全局模型
    - tasks: 所有任务的信息列表，每个元素格式为：
        {
            "task_id": ...,
            "classes": ...,
            "train_dataset": ...,
            "val_loader": ...,
            "test_loader": ...,
            "client_indices": ...,
            "client_datasets": ...,
            "client_loaders": ...
        }
    - config: 实验配置
    - save_dir: 结果保存目录
    """

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    global_model = global_model.to(device)

    use_replay = config.get("use_replay", True)
    replay_fraction = config.get("replay_fraction", 0.2)
    num_tasks = len(tasks)

    # =========================
    # 用于记录训练过程指标
    # 动态适配任意任务数
    # =========================
    history = {
        "phase": [],
        "round": [],
        "forgetting": [],
    }

    for task_idx in range(num_tasks):
        history[f"task{task_idx + 1}_val_acc"] = []
        history[f"task{task_idx + 1}_test_acc"] = []

    # 保存每个任务训练完成时的准确率
    task_reference_acc = {}

    def append_history(task_phase_name, round_idx, eval_results, forgetting_value):
        """
        将当前 round 的评估结果写入 history。
        """
        history["phase"].append(task_phase_name)
        history["round"].append(round_idx)
        history["forgetting"].append(forgetting_value)

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

        联邦场景下：
        - 每个旧任务本身已经有 client_indices
        - 这里先把旧任务所有客户端索引汇总
        - 再按 replay_fraction 抽样
        """
        replay_subsets = []
        random.seed(seed)

        for old_task in previous_tasks:
            all_indices = []
            for indices in old_task["client_indices"].values():
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
        print(f"Phase {task_idx + 1}: Federated Training on Task {task_idx + 1} {current_task['classes']}")
        print("=" * 60)

        current_client_datasets = current_task["client_datasets"]

        # =========================
        # 构造当前任务阶段训练数据
        # 第一个任务：直接使用当前任务数据
        # 后续任务：可选加入 replay
        # =========================
        if task_idx == 0:
            mixed_client_datasets = current_client_datasets
        else:
            if use_replay:
                print("\nReplay enabled.")

                previous_tasks = tasks[:task_idx]
                replay_dataset = build_replay_dataset_from_previous_tasks(
                    previous_tasks=previous_tasks,
                    replay_fraction=replay_fraction,
                    seed=config["seed"]
                )
                print(f"Replay memory size: {len(replay_dataset)}")

                # 这里简单采用“所有客户端共享同一份 replay 数据”
                mixed_client_datasets = {}
                for client_id, current_subset in current_client_datasets.items():
                    mixed_dataset = ConcatDataset([current_subset, replay_dataset])
                    mixed_client_datasets[client_id] = mixed_dataset
            else:
                print("\nReplay disabled.")
                mixed_client_datasets = current_client_datasets

        current_client_loaders = build_client_loaders(
            mixed_client_datasets,
            batch_size=config["train_batch_size"]
        )

        # =========================
        # 当前任务阶段的联邦训练轮次
        # =========================
        for round_idx in range(1, config["num_rounds"] + 1):
            print(f"\n--- Round {round_idx}/{config['num_rounds']} ---")

            # 执行一轮标准 FedAvg
            global_model = fedavg_round(
                global_model=global_model,
                client_loaders=current_client_loaders,
                device=device,
                local_epochs=config["local_epochs"],
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                verbose=True,
            )

            # 每轮聚合后，评估当前已见过的所有任务
            seen_task_ids = list(range(task_idx + 1))
            eval_results = evaluate_seen_tasks(global_model, tasks, seen_task_ids, device)

            for tid in seen_task_ids:
                print(
                    f"Task {tid + 1} Val  Acc: {eval_results[tid]['val_acc']:.4f} | "
                    f"Test Acc: {eval_results[tid]['test_acc']:.4f}"
                )

            # =========================
            # 计算平均遗忘
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

            append_history(
                task_phase_name=f"Task{task_idx + 1}",
                round_idx=round_idx,
                eval_results=eval_results,
                forgetting_value=forgetting
            )

        # =========================
        # 当前任务训练结束后，记录该任务准确率
        # =========================
        final_eval = evaluate_seen_tasks(global_model, tasks, [task_idx], device)
        task_reference_acc[task_idx] = final_eval[task_idx]["test_acc"]
        print(f"\nTask {task_idx + 1} reference acc: {task_reference_acc[task_idx]:.4f}")

    # =========================
    # 保存结果
    # =========================
    save_history_json(history, os.path.join(save_dir, "fed_continual_history.json"))
    save_continual_history_csv(history, os.path.join(save_dir, "fed_continual_history.csv"))

    experiment_name = "fed_continual_replay" if use_replay else "fed_continual_no_replay"
    plot_continual_history(history, save_dir, experiment_name=experiment_name)

    torch.save(global_model.state_dict(), os.path.join(save_dir, "fed_continual_model.pth"))

    summary = {
        "experiment_name": f'{config["dataset"]}_fed_continual',
        "use_replay": use_replay,
        "num_tasks": num_tasks,
        "num_rounds_per_phase": config["num_rounds"],
        "local_epochs": config["local_epochs"],
        "replay_fraction": replay_fraction if use_replay else 0.0,
        "task_reference_acc": task_reference_acc,
        "final_forgetting": history["forgetting"][-1],
    }

    # 记录每个任务的最终测试准确率
    for task_idx in range(num_tasks):
        summary[f"final_task{task_idx + 1}_test_acc"] = history[f"task{task_idx + 1}_test_acc"][-1]
    # 计算最终平均准确率
    final_task_accs = [
        summary[f"final_task{task_idx + 1}_test_acc"]
        for task_idx in range(num_tasks)
        if summary[f"final_task{task_idx + 1}_test_acc"] is not None
    ]
    summary["final_avg_test_acc"] = sum(final_task_accs) / len(final_task_accs)
    print(f"Final Average Test Acc: {summary['final_avg_test_acc']:.4f}")

    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {save_dir}")
    return history, summary