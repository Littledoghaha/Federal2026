"""
results_standard.py

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
    # centralized 集中式学习实验: 用 epoch
    # federated 联邦学习实验: 用 round
    if "epoch" in history:
        x_key = "epoch"
    elif "round" in history:
        x_key = "round"
    else:
        raise ValueError("History must contain 'epoch' or 'round'.")
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [x_key]
        for key in ["train_loss", "train_acc", "val_loss", "val_acc"]:
            if key in history:
                header.append(key)
        writer.writerow(header)
        num_rows = len(history[x_key])
        # 按行写入每个 num_rows （epoch/round） 的结果
        for i in range(num_rows):
            row = [history[x_key][i]]
            for key in ["train_loss", "train_acc", "val_loss", "val_acc"]:
                if key in history:
                    row.append(history[key][i])
            writer.writerow(row)
        if history.get("final_test_loss") is not None and history.get("final_test_acc") is not None:
            writer.writerow([])
            writer.writerow(["final_test_loss", "final_test_acc"])
            writer.writerow([history["final_test_loss"], history["final_test_acc"]])


def plot_history(history, save_dir, experiment_name="standard_experiment"):
    ensure_dir(save_dir)

    if "epoch" in history:
        x = history["epoch"]
        xlabel = "Epoch"
    elif "round" in history:
        x = history["round"]
        xlabel = "Round"
    else:
        raise ValueError("History must contain 'epoch' or 'round'.")

    # 1. Loss 图：优先画 train vs val 对比
    if "train_loss" in history and "val_loss" in history:
        plt.figure(figsize=(6, 4))
        # plt.plot(x, history["train_loss"], marker="o", label="Train Loss")
        # plt.plot(x, history["val_loss"], marker="s", label="Val Loss")
        plt.plot(x, history["train_loss"], linewidth=1.8, label="Train Loss") # *
        plt.plot(x, history["val_loss"], linewidth=1.8, label="Val Loss") # *
        plt.xlabel(xlabel)
        plt.ylabel("Loss")
        plt.title(f"{experiment_name} - Train vs Val Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{experiment_name}_loss.png"))
        plt.close()
    elif "val_loss" in history:
        plt.figure(figsize=(6, 4))
        # plt.plot(x, history["val_loss"], marker="o", label="Val Loss")
        plt.plot(x, history["val_loss"], linewidth=1.8, label="Val Loss") # *
        plt.xlabel(xlabel)
        plt.ylabel("Loss")
        plt.title(f"{experiment_name} - Val Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{experiment_name}_val_loss.png"))
        plt.close()

    # 2. Accuracy 图：优先画 train vs val 对比
    if "train_acc" in history and "val_acc" in history:
        plt.figure(figsize=(6, 4))
        # plt.plot(x, history["train_acc"], marker="o", label="Train Acc")
        # plt.plot(x, history["val_acc"], marker="s", label="Val Acc")
        plt.plot(x, history["train_acc"], linewidth=1.8, label="Train Acc") # *
        plt.plot(x, history["val_acc"], linewidth=1.8, label="Val Acc") # *
        plt.xlabel(xlabel)
        plt.ylabel("Accuracy")
        plt.title(f"{experiment_name} - Train vs Val Accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{experiment_name}_acc.png"))
        plt.close()
    elif "val_acc" in history:
        plt.figure(figsize=(6, 4))
        # plt.plot(x, history["val_acc"], marker="o", label="Val Acc")
        plt.plot(x, history["val_acc"], linewidth=1.8, label="Val Acc") # *
        plt.xlabel(xlabel)
        plt.ylabel("Accuracy")
        plt.title(f"{experiment_name} - Val Accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{experiment_name}_val_acc.png"))
        plt.close()