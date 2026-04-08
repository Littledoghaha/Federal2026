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
    task1_accs = history["task1_test_acc"]
    task2_accs = history["task2_test_acc"]
    forgetting_list = history["forgetting"]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase",
            "round",
            "task1_test_acc",
            "task2_test_acc",
            "forgetting"
        ])
        for phase, r, acc1, acc2, fg in zip(
            phases, rounds, task1_accs, task2_accs, forgetting_list
        ):
            writer.writerow([phase, r, acc1, acc2, fg])


def plot_continual_history(history, save_dir, experiment_name="continual_cifar10"):
    """
    绘制持续学习实验曲线：
    1. Task1 / Task2 测试准确率曲线
    2. Forgetting 曲线
    """
    ensure_dir(save_dir)
    x = list(range(1, len(history["round"]) + 1))
    labels = [f'{p}-{r}' for p, r in zip(history["phase"], history["round"])]
    task1_accs = history["task1_test_acc"]
    task2_accs = history["task2_test_acc"]
    forgetting_list = history["forgetting"]
    # None 转 nan，方便 matplotlib 跳过不画
    task2_plot = [acc if acc is not None else math.nan for acc in task2_accs]
    forgetting_plot = [fg if fg is not None else math.nan for fg in forgetting_list]
    # 1. 准确率曲线
    plt.figure(figsize=(8, 4))
    plt.plot(x, task1_accs, marker="o", label="Task 1 Test Acc")
    plt.plot(x, task2_plot, marker="s", label="Task 2 Test Acc")
    plt.xlabel("Training Stage")
    plt.ylabel("Accuracy")
    plt.title(f"{experiment_name} - Accuracy")
    plt.xticks(x, labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_acc.png"))
    plt.close()
    # 2. forgetting 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(x, forgetting_plot, marker="^", color="red", label="Forgetting")
    plt.xlabel("Training Stage")
    plt.ylabel("Forgetting")
    plt.title(f"{experiment_name} - Forgetting")
    plt.xticks(x, labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_forgetting.png"))
    plt.close()