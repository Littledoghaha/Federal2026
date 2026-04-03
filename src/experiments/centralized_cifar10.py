import os
import json
import csv
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(config_path="config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_result_dir(experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", f"{experiment_name}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def save_config_copy(config, result_dir):
    config_save_path = os.path.join(result_dir, "config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def save_history_json(history, result_dir):
    path = os.path.join(result_dir, "history.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def save_history_csv(history, result_dir):
    path = os.path.join(result_dir, "history.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_acc"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["test_loss"][i],
                history["test_acc"][i]
            ])


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / total_samples
    acc = correct / total_samples
    return avg_loss, acc


def main():
    config = load_config("E:\\federal\\src\\config.json")
    set_seed(config["seed"])

    device = torch.device(config["device"] if torch.cuda.is_available() or config["device"] == "cpu" else "cpu")

    result_dir = create_result_dir(config["experiment_name"] + "_centralized")
    save_config_copy(config, result_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["test_batch_size"],
        shuffle=False
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    num_epochs = config["num_rounds"]  # 先复用这个字段
    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "test_acc": []
    }

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"[Centralized] Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.4f}")

    total_time = time.time() - start_time

    save_history_json(history, result_dir)
    save_history_csv(history, result_dir)

    torch.save(model.state_dict(), os.path.join(result_dir, "model.pth"))

    summary = {
        "experiment_name": config["experiment_name"] + "_centralized",
        "num_epochs": num_epochs,
        "final_test_acc": history["test_acc"][-1],
        "final_test_loss": history["test_loss"][-1],
        "total_time_sec": total_time
    }
    with open(os.path.join(result_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {result_dir}")


if __name__ == "__main__":
    main()