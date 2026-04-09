"""
results_continual.py

负责持续学习实验结果的保存与可视化：
1. 保存 continual history 到 csv
2. 绘制 Task1 / Task2 测试准确率曲线
3. 绘制 forgetting 曲线
"""

import os
import csv
import math
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _build_sparse_xticks(labels, max_ticks=12):
    """
    稀疏显示横坐标，避免标签挤在一起。
    max_ticks: 最多显示多少个横坐标标签
    返回:
        tick_positions: matplotlib 用的横坐标位置
        tick_labels:    对应显示的标签文本
    """
    n = len(labels)
    if n <= max_ticks:
        # 点数不多就全显示
        return list(range(1, n + 1)), labels

    # 自动计算步长，比如 200 个点、最多显示 12 个标签，则每隔 17 个左右显示一次
    step = math.ceil(n / max_ticks)
    tick_positions = list(range(1, n + 1, step))
    tick_labels = [labels[i - 1] for i in tick_positions]

    # 保证最后一个点也显示出来
    if tick_positions[-1] != n:
        tick_positions.append(n)
        tick_labels.append(labels[-1])

    return tick_positions, tick_labels

def save_continual_history_csv(history, save_path):
    """
    保存持续学习实验 history 为 CSV。
    期望 history 结构：
    {
        "phase": [...],
        "round": [...],
        "task1_test_acc": [...],
        "task2_test_acc": [...],
        "forgetting": [...],
    }
    """
    ensure_dir(os.path.dirname(save_path))
    phases = history["phase"]
    rounds = history["round"]
    task1_val_accs = history["task1_val_acc"] # *
    task1_test_accs = history["task1_test_acc"]
    task2_val_accs = history["task2_val_acc"] # *
    task2_test_accs = history["task2_test_acc"]
    forgetting_list = history["forgetting"]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase",
            "round",
            "task1_val_acc", # *
            "task1_test_acc",
            "task2_val_acc", # *
            "task2_test_acc",
            "forgetting"
        ])
        # for phase, r, acc1, acc2, fg in zip(
        #     phases, rounds, task1_accs, task2_accs, forgetting_list
        # ):
        #     writer.writerow([phase, r, acc1, acc2, fg])
        for phase, r, t1val, t1test, t2val, t2test, fg in zip(
            phases, rounds,
            task1_val_accs, task1_test_accs,
            task2_val_accs, task2_test_accs,
            forgetting_list
        ):
            writer.writerow([phase, r, t1val, t1test, t2val, t2test, fg])


def plot_continual_history(history, save_dir, experiment_name="continual_cifar10"):
    """
    绘制持续学习实验曲线：
    1. Task1 / Task2 测试准确率曲线
    2. Forgetting 曲线
    """
    ensure_dir(save_dir)
    x = list(range(1, len(history["round"]) + 1))
    labels = [f'{p}-{r}' for p, r in zip(history["phase"], history["round"])]
    # 自动稀疏显示横坐标，避免过密
    tick_positions, tick_labels = _build_sparse_xticks(labels, max_ticks=12) 
    # 读取精度曲线
    task1_val_accs = history["task1_val_acc"] # *
    task1_test_accs = history["task1_test_acc"]
    task2_val_accs = history["task2_val_acc"] # *
    task2_test_accs = history["task2_test_acc"]
    forgetting_list = history["forgetting"]
    # None 转 nan，方便 matplotlib 跳过不画
    task1_val_plot = [acc if acc is not None else math.nan for acc in task1_val_accs]
    task1_test_plot = [acc if acc is not None else math.nan for acc in task1_test_accs]
    task2_val_plot = [acc if acc is not None else math.nan for acc in task2_val_accs]
    task2_test_plot = [acc if acc is not None else math.nan for acc in task2_test_accs]
    # task2_plot = [acc if acc is not None else math.nan for acc in task2_test_accs]
    forgetting_plot = [fg if fg is not None else math.nan for fg in forgetting_list]
    # 1. val / test accuracy 曲线（单实验一张图，共4条线）
    plt.figure(figsize=(10, 5))
    plt.plot(x, task1_val_plot, marker="o", markersize=3, linewidth=1.5, label="Task1 Val Acc")
    plt.plot(x, task1_test_plot, marker="o", markersize=3, linewidth=1.5, linestyle="--", label="Task1 Test Acc")
    plt.plot(x, task2_val_plot, marker="s", markersize=3, linewidth=1.5, label="Task2 Val Acc")
    plt.plot(x, task2_test_plot, marker="s", markersize=3, linewidth=1.5, linestyle="--", label="Task2 Test Acc")
    plt.xlabel("Training Stage")
    plt.ylabel("Accuracy")
    plt.title(f"{experiment_name} - Val/Test Accuracy")
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_val_test_acc.png"))
    plt.close()
    # 1. 准确率曲线
    # plt.figure(figsize=(8, 4))
    # plt.plot(x, task1_test_accs, marker="o", label="Task 1 Test Acc")
    # plt.plot(x, task2_plot, marker="s", label="Task 2 Test Acc")
    # plt.xlabel("Training Stage")
    # plt.ylabel("Accuracy")
    # plt.title(f"{experiment_name} - Accuracy")
    # # plt.xticks(x, labels, rotation=45)
    # plt.xticks(tick_positions, tick_labels, rotation=45)
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f"{experiment_name}_acc.png"))
    # plt.close()
    # 2. forgetting 曲线
    # plt.figure(figsize=(8, 4))
    # plt.plot(x, forgetting_plot, marker="^", color="red", label="Forgetting")
    # plt.xlabel("Training Stage")
    # plt.ylabel("Forgetting")
    # plt.title(f"{experiment_name} - Forgetting")
    # # plt.xticks(x, labels, rotation=45)
    # plt.xticks(tick_positions, tick_labels, rotation=45)
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f"{experiment_name}_forgetting.png"))
    # plt.close()
    # 只绘制 Task2 阶段的 forgetting，避免左半边全是空白
    task2_indices = [i for i, p in enumerate(history["phase"]) if p == "Task2"]
    if len(task2_indices) > 0:
        x_fg = [i + 1 for i in task2_indices]
        fg_labels = [labels[i] for i in task2_indices]
        fg_values = [forgetting_list[i] for i in task2_indices]
        fg_plot = [fg if fg is not None else math.nan for fg in fg_values]
        fg_tick_positions, fg_tick_labels = _build_sparse_xticks(fg_labels, max_ticks=12)
        # 注意：_build_sparse_xticks 返回的是从 1 开始的相对位置，
        # 这里需要映射回真正的 x 轴位置
        fg_tick_positions = [x_fg[i - 1] for i in fg_tick_positions]
        plt.figure(figsize=(8, 4))
        plt.plot(x_fg, fg_plot, marker="^", color="red", label="Forgetting")
        plt.xlabel("Training Stage")
        plt.ylabel("Forgetting")
        plt.title(f"{experiment_name} - Forgetting")
        plt.xticks(fg_tick_positions, fg_tick_labels, rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{experiment_name}_forgetting.png"))
        plt.close()

def plot_continual_comparison(history_no_replay, history_replay, save_dir,
                              experiment_name="continual_cifar10_comparison"):
    """
    将 no replay 和 replay 的结果画到同一张图中进行对比。
    包括：
    1. Task1 Test Accuracy 对比
    2. Task2 Test Accuracy 对比
    3. Forgetting 对比
    """
    ensure_dir(save_dir)

    x = list(range(1, len(history_no_replay["round"]) + 1))
    labels = [f'{p}-{r}' for p, r in zip(history_no_replay["phase"], history_no_replay["round"])]
    tick_positions, tick_labels = _build_sparse_xticks(labels, max_ticks=12)

    no_task1 = history_no_replay["task1_test_acc"]
    no_task2 = [acc if acc is not None else math.nan for acc in history_no_replay["task2_test_acc"]]
    no_forgetting = [fg if fg is not None else math.nan for fg in history_no_replay["forgetting"]]

    re_task1 = history_replay["task1_test_acc"]
    re_task2 = [acc if acc is not None else math.nan for acc in history_replay["task2_test_acc"]]
    re_forgetting = [fg if fg is not None else math.nan for fg in history_replay["forgetting"]]

    # 1. Accuracy comparison
    plt.figure(figsize=(9, 5))
    plt.plot(x, no_task1, marker="o", label="Task1 Acc (No Replay)")
    plt.plot(x, re_task1, marker="o", linestyle="--", label="Task1 Acc (Replay)")
    plt.plot(x, no_task2, marker="s", label="Task2 Acc (No Replay)")
    plt.plot(x, re_task2, marker="s", linestyle="--", label="Task2 Acc (Replay)")
    plt.xlabel("Training Stage")
    plt.ylabel("Accuracy")
    plt.title(f"{experiment_name} - Accuracy Comparison")
    # plt.xticks(x, labels, rotation=45)
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_acc_comparison.png"))
    plt.close()

    # 2. Forgetting comparison
    # plt.figure(figsize=(9, 5))
    # plt.plot(x, no_forgetting, marker="^", color="red", label="Forgetting (No Replay)")
    # plt.plot(x, re_forgetting, marker="^", color="green", linestyle="--", label="Forgetting (Replay)")
    # plt.xlabel("Training Stage")
    # plt.ylabel("Forgetting")
    # plt.title(f"{experiment_name} - Forgetting Comparison")
    # # plt.xticks(x, labels, rotation=45)
    # plt.xticks(tick_positions, tick_labels, rotation=45)
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f"{experiment_name}_forgetting_comparison.png"))
    # plt.close()
    # 只画 Task2 阶段的 forgetting，对比图更干净
    task2_indices = [i for i, p in enumerate(history_no_replay["phase"]) if p == "Task2"]

    if len(task2_indices) > 0:
        x_fg = [i + 1 for i in task2_indices]
        fg_labels = [labels[i] for i in task2_indices]

        no_fg = [history_no_replay["forgetting"][i] for i in task2_indices]
        re_fg = [history_replay["forgetting"][i] for i in task2_indices]

        no_fg_plot = [fg if fg is not None else math.nan for fg in no_fg]
        re_fg_plot = [fg if fg is not None else math.nan for fg in re_fg]

        fg_tick_positions, fg_tick_labels = _build_sparse_xticks(fg_labels, max_ticks=12)
        fg_tick_positions = [x_fg[i - 1] for i in fg_tick_positions]

        plt.figure(figsize=(9, 5))
        plt.plot(x_fg, no_fg_plot, marker="^", color="red", label="Forgetting (No Replay)")
        plt.plot(x_fg, re_fg_plot, marker="^", color="green", linestyle="--", label="Forgetting (Replay)")
        plt.xlabel("Training Stage")
        plt.ylabel("Forgetting")
        plt.title(f"{experiment_name} - Forgetting Comparison")
        plt.xticks(fg_tick_positions, fg_tick_labels, rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{experiment_name}_forgetting_comparison.png"))
        plt.close()