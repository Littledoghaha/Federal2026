"""
普通联邦学习（fedavg）在 CIFAR-10 上的 baseline 实验。
"""

import torch
import os
import json
from datasets.data_loader import prepare_federated_dataloaders
from core.model import SimpleCNN
from core.fedavg import run_fedavg
from utils.results_standard import save_history_json, save_history_csv, plot_history


from datetime import datetime


def run(config, save_dir):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data_bundle = prepare_federated_dataloaders(
        root="data",
        num_clients=config["num_clients"],
        train_batch_size=config["train_batch_size"],
        test_batch_size=config["test_batch_size"], # train_batch_size 用于客户端本地训练；test_batch_size 同时用于验证集和测试集评估
        seed=config["seed"],
        num_workers=0,
    )
    cifar_client_loaders = data_bundle["cifar"]["client_loaders"]
    cifar_val_loader = data_bundle["cifar"]["val_loader"]
    cifar_test_loader = data_bundle["cifar"]["test_loader"]
    print("\nCIFAR-10 client sizes:")
    for client_id, loader in cifar_client_loaders.items():
        print(f"client {client_id}: {len(loader.dataset)} samples")
    global_model = SimpleCNN(num_classes=10).to(device)
    global_model, history = run_fedavg(
        global_model=global_model,
        client_loaders=cifar_client_loaders,
        val_loader=cifar_val_loader,
        test_loader=cifar_test_loader,
        device=device,
        num_rounds=config["num_rounds"],
        local_epochs=config["local_epochs"],
        lr=config["lr"],
        weight_decay=config["weight_decay"], # 正则化参数，本质上是为了防止模型过拟合
        verbose=True,
    )

    model_save_path = os.path.join(save_dir, "global_model_final.pth")
    torch.save(global_model.state_dict(), model_save_path)
    save_history_json(history, os.path.join(save_dir, "history.json"))
    save_history_csv(history, os.path.join(save_dir, "history.csv"))
    plot_history(history, save_dir, f'{config["dataset"]}_federated')
    summary = {
        "experiment_name": f'{config["dataset"]}_federated',
        "num_rounds": config["num_rounds"],
        "local_epochs": config["local_epochs"],
        "num_clients": config["num_clients"],
        "final_val_acc": history["val_acc"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_test_acc": history["final_test_acc"],
        "final_test_loss": history["final_test_loss"],
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nDone.")
    print("val_acc history:", [round(x, 4) for x in history["val_acc"]])
    print("final_test_acc:", round(history["final_test_acc"], 4))
    print(f"Results saved to: {save_dir}")
