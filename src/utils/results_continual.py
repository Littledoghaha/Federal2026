"""
results_continual.py

负责持续学习实验结果的保存与可视化：
1. 保存 continual history 到 csv
2. 绘制各任务的 Val/Test Accuracy 曲线
3. 绘制 forgetting 曲线
4. 绘制 no replay / replay 对比图

本版本已从“固定支持 task1/task2”改为“自动识别任意 task 数”。
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

    # 自动计算步长，比如 200 个点、最多显示 12 个标签，则每隔若干个显示一次
    step = math.ceil(n / max_ticks)
    tick_positions = list(range(1, n + 1, step))
    tick_labels = [labels[i - 1] for i in tick_positions]

    # 保证最后一个点也显示出来
    if tick_positions[-1] != n:
        tick_positions.append(n)
        tick_labels.append(labels[-1])

    return tick_positions, tick_labels


def _get_task_ids_from_history(history):
    """
    从 history 的 key 中自动解析有哪些任务。

    例如 history 中如果有：
    - task1_test_acc
    - task2_test_acc
    - task3_test_acc

    则返回：
    [1, 2, 3]
    """
    task_ids = []
    for key in history.keys():
        if key.startswith("task") and key.endswith("_test_acc"):
            task_num = int(key.replace("task", "").replace("_test_acc", ""))
            task_ids.append(task_num)

    task_ids.sort()
    return task_ids


def save_continual_history_csv(history, save_path):
    """
    保存持续学习实验 history 为 CSV。

    说明：
    - 本版本支持任意 task 数
    - 会自动把 task1/task2/task3/... 的 val/test acc 都写入 CSV
    """
    ensure_dir(os.path.dirname(save_path))

    task_ids = _get_task_ids_from_history(history)

    header = ["phase", "round"]
    for tid in task_ids:
        header.append(f"task{tid}_val_acc")
        header.append(f"task{tid}_test_acc")
    header.append("forgetting")

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        num_rows = len(history["round"])
        for i in range(num_rows):
            row = [history["phase"][i], history["round"][i]]

            for tid in task_ids:
                row.append(history[f"task{tid}_val_acc"][i])
                row.append(history[f"task{tid}_test_acc"][i])

            row.append(history["forgetting"][i])
            writer.writerow(row)


def plot_continual_history(history, save_dir, experiment_name="continual_cifar10"):
    """
    绘制持续学习实验曲线：
    1. 各任务的 Val/Test Accuracy 曲线
    2. Forgetting 曲线

    说明：
    - 自动适配任意 task 数
    - 例如 2 task、3 task、4 task 都可以画
    """
    ensure_dir(save_dir)

    x = list(range(1, len(history["round"]) + 1))
    labels = [f'{p}-{r}' for p, r in zip(history["phase"], history["round"])]
    tick_positions, tick_labels = _build_sparse_xticks(labels, max_ticks=12)

    task_ids = _get_task_ids_from_history(history)

    # =========================
    # 1. 各任务 Val/Test Accuracy 曲线
    # =========================
    plt.figure(figsize=(10, 5))

    # 准备一些不同的 marker，避免多任务时完全一样
    markers = ["o", "s", "^", "D", "x", "*"]

    for idx, tid in enumerate(task_ids):
        val_key = f"task{tid}_val_acc"
        test_key = f"task{tid}_test_acc"

        val_plot = [acc if acc is not None else math.nan for acc in history[val_key]]
        test_plot = [acc if acc is not None else math.nan for acc in history[test_key]]

        marker = markers[idx % len(markers)]

        plt.plot(
            x,
            val_plot,
            marker=marker,
            markersize=3,
            linewidth=1.5,
            label=f"Task{tid} Val Acc"
        )
        plt.plot(
            x,
            test_plot,
            marker=marker,
            markersize=3,
            linewidth=1.5,
            linestyle="--",
            label=f"Task{tid} Test Acc"
        )

    plt.xlabel("Training Stage")
    plt.ylabel("Accuracy")
    plt.title(f"{experiment_name} - Val/Test Accuracy")
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_val_test_acc.png"))
    plt.close()

    # =========================
    # 2. Forgetting 曲线
    # 只绘制有 forgetting 值的部分，避免前期全是空白
    # =========================
    valid_fg_indices = [i for i, v in enumerate(history["forgetting"]) if v is not None]

    if len(valid_fg_indices) > 0:
        x_fg = [i + 1 for i in valid_fg_indices]
        fg_labels = [labels[i] for i in valid_fg_indices]
        fg_values = [history["forgetting"][i] for i in valid_fg_indices]
        fg_plot = [fg if fg is not None else math.nan for fg in fg_values]

        fg_tick_positions, fg_tick_labels = _build_sparse_xticks(fg_labels, max_ticks=12)

        # _build_sparse_xticks 返回的是局部相对位置，这里映射回真正 x 轴位置
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
    1. 各任务 Test Accuracy 对比
    2. Forgetting 对比

    说明：
    - 自动适配任意 task 数
    """
    ensure_dir(save_dir)

    x = list(range(1, len(history_no_replay["round"]) + 1))
    labels = [f'{p}-{r}' for p, r in zip(history_no_replay["phase"], history_no_replay["round"])]
    tick_positions, tick_labels = _build_sparse_xticks(labels, max_ticks=12)

    task_ids = _get_task_ids_from_history(history_no_replay)

    # =========================
    # 1. Accuracy comparison
    # 这里只画 test acc，对比更直观
    # =========================
    plt.figure(figsize=(10, 5))

    markers = ["o", "s", "^", "D", "x", "*"]

    for idx, tid in enumerate(task_ids):
        marker = markers[idx % len(markers)]

        no_test = [
            acc if acc is not None else math.nan
            for acc in history_no_replay[f"task{tid}_test_acc"]
        ]
        re_test = [
            acc if acc is not None else math.nan
            for acc in history_replay[f"task{tid}_test_acc"]
        ]

        plt.plot(
            x,
            no_test,
            marker=marker,
            label=f"Task{tid} Test (No Replay)"
        )
        plt.plot(
            x,
            re_test,
            marker=marker,
            linestyle="--",
            label=f"Task{tid} Test (Replay)"
        )

    plt.xlabel("Training Stage")
    plt.ylabel("Accuracy")
    plt.title(f"{experiment_name} - Test Accuracy Comparison")
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_acc_comparison.png"))
    plt.close()

    # =========================
    # 2. Forgetting comparison
    # 只画有 forgetting 的阶段
    # =========================
    valid_fg_indices = [i for i, v in enumerate(history_no_replay["forgetting"]) if v is not None]

    if len(valid_fg_indices) > 0:
        x_fg = [i + 1 for i in valid_fg_indices]
        fg_labels = [labels[i] for i in valid_fg_indices]

        no_fg = [history_no_replay["forgetting"][i] for i in valid_fg_indices]
        re_fg = [history_replay["forgetting"][i] for i in valid_fg_indices]

        fg_tick_positions, fg_tick_labels = _build_sparse_xticks(fg_labels, max_ticks=12)
        fg_tick_positions = [x_fg[i - 1] for i in fg_tick_positions]

        plt.figure(figsize=(9, 5))
        plt.plot(x_fg, no_fg, marker="^", color="red", label="Forgetting (No Replay)")
        plt.plot(x_fg, re_fg, marker="^", color="green", linestyle="--", label="Forgetting (Replay)")
        plt.xlabel("Training Stage")
        plt.ylabel("Forgetting")
        plt.title(f"{experiment_name} - Forgetting Comparison")
        plt.xticks(fg_tick_positions, fg_tick_labels, rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{experiment_name}_forgetting_comparison.png"))
        plt.close()