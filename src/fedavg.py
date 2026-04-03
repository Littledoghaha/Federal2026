"""
FedAvg 训练流程。

负责：

复制全局模型到每个 client
每个 client 本地训练
收集参数
做加权平均
每轮联邦后做测试

当前版本默认：
1. 所有客户端都参与
2. 按各客户端样本数加权平均
"""

import copy
import torch

from train_eval import train_local, evaluate


def average_state_dicts(state_dicts, weights):
    total_weight = sum(weights)
    avg_state = copy.deepcopy(state_dicts[0])

    for key in avg_state.keys():
        if torch.is_floating_point(avg_state[key]):
            avg_state[key] = torch.zeros_like(avg_state[key])
            for state, weight in zip(state_dicts, weights):
                avg_state[key] += state[key] * (weight / total_weight)
                # ***********************************
                # *           权重平均               *
                # ***********************************
                # 确认联邦平均算法的运行模式
                
        else:
            # 非浮点 tensor（当前模型基本不会碰到）
            avg_state[key] = state_dicts[0][key].clone()

    return avg_state


def fedavg_round(
    global_model,
    client_loaders,
    device,
    local_epochs=1,
    lr=1e-3,
    weight_decay=0.0,
    verbose=True,
):
    local_states = []
    local_weights = []

    for client_id, train_loader in client_loaders.items():
        local_model = copy.deepcopy(global_model).to(device)

        local_model, history = train_local(
            model=local_model,
            train_loader=train_loader,
            device=device,
            epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            verbose=False
        )

        local_states.append(copy.deepcopy(local_model.state_dict()))
        local_weights.append(len(train_loader.dataset))

        if verbose:
            train_loss = history["train_loss"][-1]
            train_acc = history["train_acc"][-1]
            print(
                f"client {client_id} "
                f"- samples: {len(train_loader.dataset)} "
                f"- train_loss: {train_loss:.4f} "
                f"- train_acc: {train_acc:.4f}"
            )

    new_global_state = average_state_dicts(local_states, local_weights)
    global_model.load_state_dict(new_global_state)

    return global_model


def run_fedavg(
    global_model,
    client_loaders,
    test_loader,
    device,
    num_rounds=3,
    local_epochs=1,
    lr=1e-3,
    weight_decay=0.0,
    verbose=True,
):
    history = {
        "round": [],
        "test_loss": [],
        "test_acc": [],
    }

    # round 0：未训练前先测一次
    test_loss, test_acc = evaluate(global_model, test_loader, device)
    print(f"round 0 - test_loss: {test_loss:.4f} - test_acc: {test_acc:.4f}")

    history["round"].append(0)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)

    for round_idx in range(1, num_rounds + 1):
        print(f"\n=== Federated Round {round_idx}/{num_rounds} ===")

        global_model = fedavg_round(
            global_model=global_model,
            client_loaders=client_loaders,
            device=device,
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            verbose=verbose,
        )

        test_loss, test_acc = evaluate(global_model, test_loader, device)
        print(f"round {round_idx} - test_loss: {test_loss:.4f} - test_acc: {test_acc:.4f}")

        history["round"].append(round_idx)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

    return global_model, history