"""
训练与评估模块。
包含：
1. 单个 epoch 的训练
2. 本地训练（多个 epoch）
3. 测试集评估
"""

import torch
import torch.nn as nn


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def train_local(
    model,
    train_loader,
    device,
    epochs=1,
    lr=1e-3,
    weight_decay=0.0,
    verbose=True
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    history = {
        "train_loss": [],
        "train_acc": []
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if verbose:
            print(
                f"Epoch {epoch}/{epochs} "
                f"- train_loss: {train_loss:.4f} "
                f"- train_acc: {train_acc:.4f}"
            )

    return model, history


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc