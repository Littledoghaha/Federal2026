import os
import json
import random
import copy
import torch
from torch.utils.data import Subset, ConcatDataset

from core.fedavg import fedavg_round
from core.train_eval import evaluate
from datasets.client_dataloader import build_client_loaders
from utils.results_standard import save_history_json
from utils.results_continual import save_continual_history_csv, plot_continual_history


def run_federated_continual_learning(
    global_model,
    task1_client_datasets,
    task1_client_indices,
    task1_test_loader,
    task2_client_datasets,
    task2_test_loader,
    config,
    save_dir
):
    def set_seed(seed=42):
        random.seed(seed)
        torch.manual_seed(seed)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    global_model = global_model.to(device)

    use_replay = config.get("use_replay", True)
    replay_fraction = config.get("replay_fraction", 0.2)

    task1_client_loaders = build_client_loaders(
        task1_client_datasets,
        batch_size=config["train_batch_size"]
    )

    history = {
        "phase": [],
        "round": [],
        "task1_test_acc": [],
        "task2_test_acc": [],
        "forgetting": [],
    }

    print("\n" + "=" * 60)
    print("Phase 1: Federated Training on Task 1")
    print("=" * 60)

    # round 0 可选
    _, task1_acc = evaluate(global_model, task1_test_loader, device)
    print(f"Phase1 round 0 - Task1 Test Acc: {task1_acc:.4f}")

    for round_idx in range(1, config["num_rounds"] + 1):
        print(f"\n--- Phase1 Round {round_idx}/{config['num_rounds']} ---")

        global_model = fedavg_round(
            global_model=global_model,
            client_loaders=task1_client_loaders,
            device=device,
            local_epochs=config["local_epochs"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            verbose=True,
        )

        _, task1_acc = evaluate(global_model, task1_test_loader, device)
        print(f"Task 1 Test Acc: {task1_acc:.4f}")

        history["phase"].append("Task1")
        history["round"].append(round_idx)
        history["task1_test_acc"].append(task1_acc)
        history["task2_test_acc"].append(None)
        history["forgetting"].append(None)

    task1_reference_acc = max(history["task1_test_acc"])
    print(f"\nTask 1 reference acc: {task1_reference_acc:.4f}")

    # 构造 replay
    if use_replay:
        print("\nReplay enabled.")
        all_task1_indices = []
        for indices in task1_client_indices.values():
            all_task1_indices.extend(indices)

        replay_size = max(int(len(all_task1_indices) * replay_fraction), 1)
        replay_indices = random.sample(all_task1_indices, replay_size)

        all_task1_dataset = None
        for ds in task1_client_datasets.values():
            all_task1_dataset = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
            break

        task1_replay_dataset = Subset(all_task1_dataset, replay_indices)

        task2_client_mixed_datasets = {}
        for client_id, task2_subset in task2_client_datasets.items():
            mixed_dataset = ConcatDataset([task2_subset, task1_replay_dataset])
            task2_client_mixed_datasets[client_id] = mixed_dataset
    else:
        print("\nReplay disabled.")
        task2_client_mixed_datasets = task2_client_datasets

    task2_client_loaders = build_client_loaders(
        task2_client_mixed_datasets,
        batch_size=config["train_batch_size"]
    )

    print("\n" + "=" * 60)
    print("Phase 2: Federated Continual Training on Task 2")
    print("=" * 60)

    for round_idx in range(1, config["num_rounds"] + 1):
        print(f"\n--- Phase2 Round {round_idx}/{config['num_rounds']} ---")

        global_model = fedavg_round(
            global_model=global_model,
            client_loaders=task2_client_loaders,
            device=device,
            local_epochs=config["local_epochs"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            verbose=True,
        )

        _, task1_acc = evaluate(global_model, task1_test_loader, device)
        _, task2_acc = evaluate(global_model, task2_test_loader, device)
        forgetting = task1_reference_acc - task1_acc

        print(f"Task 1 Test Acc: {task1_acc:.4f}")
        print(f"Task 2 Test Acc: {task2_acc:.4f}")
        print(f"Forgetting: {forgetting:.4f}")

        history["phase"].append("Task2")
        history["round"].append(round_idx)
        history["task1_test_acc"].append(task1_acc)
        history["task2_test_acc"].append(task2_acc)
        history["forgetting"].append(forgetting)

    save_history_json(history, os.path.join(save_dir, "fed_continual_history.json"))
    save_continual_history_csv(history, os.path.join(save_dir, "fed_continual_history.csv"))

    experiment_name = "fed_continual_replay" if use_replay else "fed_continual_no_replay"
    plot_continual_history(history, save_dir, experiment_name=experiment_name)

    torch.save(global_model.state_dict(), os.path.join(save_dir, "fed_continual_model.pth"))

    summary = {
        "experiment_name": f'{config["dataset"]}_fed_continual',
        "use_replay": use_replay,
        "num_rounds_per_phase": config["num_rounds"],
        "local_epochs": config["local_epochs"],
        "replay_fraction": replay_fraction if use_replay else 0.0,
        "task1_reference_acc": task1_reference_acc,
        "final_task1_test_acc": history["task1_test_acc"][-1],
        "final_task2_test_acc": history["task2_test_acc"][-1],
        "final_forgetting": history["forgetting"][-1],
    }

    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {save_dir}")
    return history, summary