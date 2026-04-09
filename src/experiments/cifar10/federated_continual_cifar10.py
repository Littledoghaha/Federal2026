import torch
import random
from types import SimpleNamespace

from datasets.data_loader import prepare_federated_continual_dataloaders
from core.model import SimpleCNN
from core.fed_continual import run_federated_continual_learning


def run_federated_continual(config, save_dir):
    def set_seed(seed=42):
        random.seed(seed)
        torch.manual_seed(seed)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_args = SimpleNamespace(
        lambda_l1=config.get("lambda_l1", 1e-4),
        lambda_mask=config.get("lambda_mask", 1e-4),
        device=device,
    )

    task_data = prepare_federated_continual_dataloaders(
        dataset="cifar10",
        num_clients=config["num_clients"],
        num_tasks=2,
        train_batch_size=config["train_batch_size"],
        test_batch_size=config["test_batch_size"],
        seed=config["seed"],
        num_workers=0,
        val_ratio=0.1,
        task_classes=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    )

    task1_data = task_data["task0"]
    task2_data = task_data["task1"]

    print("\nTask split summary:")
    print(f"Task 1 train samples: {len(task1_data['train_dataset'])}")
    print(f"Task 2 train samples: {len(task2_data['train_dataset'])}")

    for cid, ds in task1_data["client_datasets"].items():
        print(f"Task1 client {cid}: {len(ds)} samples")
    for cid, ds in task2_data["client_datasets"].items():
        print(f"Task2 client {cid}: {len(ds)} samples")

    model = SimpleCNN(args=model_args, num_classes=10, in_channels=3).to(device)

    history, summary = run_federated_continual_learning(
        global_model=model,
        task1_client_datasets=task1_data["client_datasets"],
        task1_client_indices=task1_data["client_indices"],
        task1_test_loader=task1_data["test_loader"],
        task2_client_datasets=task2_data["client_datasets"],
        task2_test_loader=task2_data["test_loader"],
        config=config,
        save_dir=save_dir,
    )

    return history, summary