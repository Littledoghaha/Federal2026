"""
results_utils.py

负责：
1. 保存训练 history 到 json
2. 保存训练 history 到 csv
3. 绘制测试 loss / acc 曲线
"""

import os
import json
import csv
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_history_json(history, save_path):
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)


def save_history_csv(history, save_path):
    ensure_dir(os.path.dirname(save_path))

    rounds = history["round"]
    test_losses = history["test_loss"]
    test_accs = history["test_acc"]

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "test_loss", "test_acc"])

        for r, loss, acc in zip(rounds, test_losses, test_accs):
            writer.writerow([r, loss, acc])


def plot_history(history, save_dir, experiment_name="cifar10_federated"):
    ensure_dir(save_dir)

    rounds = history["round"]
    test_losses = history["test_loss"]
    test_accs = history["test_acc"]

    # 画 test loss 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(rounds, test_losses, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Test Loss")
    plt.title(f"{experiment_name} - Test Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_test_loss.png"))
    plt.close()

    # 画 test acc 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(rounds, test_accs, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.title(f"{experiment_name} - Test Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_test_acc.png"))
    plt.close()