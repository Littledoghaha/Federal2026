"""
集中式学习实验（CIFAR-10）

1. 在单机上直接使用完整 CIFAR-10 训练集训练模型
2. 作为联邦学习 / 持续学习实验的 baseline（基线对照）

"""

import os
import json
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from core.model import SimpleCNN

def set_seed(seed=42):
    """
    设置随机种子，保证实验尽量可复现。
    为什么要设置？
    - random: Python 自带随机库
    - numpy: NumPy 的随机数
    - torch: PyTorch 的随机数
    这样每次运行时，数据顺序、初始化等会更稳定一些。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_history_json(history, result_dir):
    # 将训练历史保存为 JSON 文件
    path = os.path.join(result_dir, "history.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def save_history_csv(history, result_dir):
    # 将训练历史保存为 CSV 文件
    path = os.path.join(result_dir, "history.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_acc"])
        # 按行写入每个 epoch 的结果
        for i in range(len(history["epoch"])):
            writer.writerow(
                [
                    history["epoch"][i],
                    history["train_loss"][i],
                    history["test_loss"][i],
                    history["test_acc"][i],
                ]
            )


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练一个 epoch。
    参数说明：
    - model: 模型
    - train_loader: 训练数据加载器
    - optimizer: 优化器（这里是 Adam）
    - criterion: 损失函数（这里是交叉熵）
    - device: 训练设备（cpu 或 cuda）
    返回：
    - 当前 epoch 的平均训练损失
    """
    # 训练一个 epoch 并返回当前epoch的平均训练损失
    model.train()  # 切换到训练模式
    total_loss = 0.0
    total_samples = 0

    for images, labels in train_loader:
        # 把数据放到设备上
        images = images.to(device)
        labels = labels.to(device)
        # 1. 清空上一轮梯度
        optimizer.zero_grad()
        # 2. 前向传播
        outputs = model(images)
        # 3. 计算损失
        loss = criterion(outputs, labels)
        # 4. 反向传播
        loss.backward()
        # 5. 更新参数
        optimizer.step()
        # 统计当前 batch 的损失和样本数
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    # 返回整个 epoch 的平均损失
    return total_loss / total_samples


def evaluate(model, test_loader, criterion, device):
    """
    在测试集上评估模型。
    返回：
    - avg_loss: 平均测试损失
    - acc: 测试准确率
    """
    model.eval()  # 切换到评估模式
    total_loss = 0.0
    correct = 0
    total_samples = 0
    # 测试阶段不需要计算梯度，可以节省显存和时间
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            # 取每个样本预测分数最大的类别作为预测结果
            preds = outputs.argmax(dim=1)
            # 统计预测正确的数量
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / total_samples
    acc = correct / total_samples
    return avg_loss, acc


def run_centralized(config, result_dir):
    # 1. 设置随机种子和设备
    set_seed(config["seed"])
    device = torch.device(
        config["device"]
        if torch.cuda.is_available() or config["device"] == "cpu"
        else "cpu"
    )
    print(f"Device: {device}")

    # 2. 定义数据预处理
    # 这里只做了最基础的 ToTensor()，即把图像转成张量
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    # 3. 加载 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    # 4. 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config["train_batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["test_batch_size"], shuffle=False
    )
    # 5. 初始化模型、损失函数和优化器
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()  # 采用交叉熵损失函数，适用于分类任务
    optimizer = optim.Adam(  # 采用 Adam 优化器
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    num_epochs = config["num_rounds"]  # 设置集中式学习轮次，跟联邦学习的轮次一致
    history = {"epoch": [], "train_loss": [], "test_loss": [], "test_acc": []}

    start_time = time.time()

    # 6. 开始训练
    for epoch in range(1, num_epochs + 1):
        # 训练一个 epoch
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        # 在测试集上评估
        test_loss, test_acc = evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )

        # 保存当前 epoch 的结果
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"[Centralized] Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    total_time = time.time() - start_time

    save_history_json(history, result_dir)
    save_history_csv(history, result_dir)
    # 保存模型参数
    torch.save(model.state_dict(), os.path.join(result_dir, "model.pth"))

    summary = {
        "experiment_name": f'{config["dataset"]}_centralized',
        "num_epochs": num_epochs,
        "final_test_acc": history["test_acc"][-1],
        "final_test_loss": history["test_loss"][-1],
        "total_time_sec": total_time,
    }

    with open(os.path.join(result_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {result_dir}")

