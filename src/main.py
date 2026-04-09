"""
项目总入口。运行三个实验：
1. 集中式学习（baseline）
2. 联邦学习（baseline）
3. 持续学习（观察遗忘）
"""
import sys
import os
from experiments.cifar10.fedavg_cifar10 import run as run_fedavg
# load_config, create_result_dir, save_config_copy
from experiments.cifar10.centralized_cifar10 import run_centralized as run_centralized
from experiments.cifar10.continual_cifar10 import run_continual
from utils.config import load_config,create_result_dir,save_config_copy
from utils.results_continual import plot_continual_comparison
import copy
from experiments.cifar10.federated_continual_cifar10 import run_federated_continual

def main():
    config = load_config("E:\\federal\\src\\config.json")
    dataset = config["dataset"]
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if mode in ["centralized", "all"]:
        print("\n" + "="*60)
        print("Experiment 1: Centralized Learning")
        print("="*60)
        result_dir = create_result_dir(dataset, "centralized")
        save_config_copy(config, result_dir)
        run_centralized(config, result_dir)
    
    if mode in ["federated", "all"]:
        print("\n" + "="*60)
        print("Experiment 2: Federated Learning")
        print("="*60)
        result_dir = create_result_dir(dataset, "federated")
        save_config_copy(config, result_dir)
        run_fedavg(config, result_dir)
    
    if mode in ["continual", "all"]:
        print("\n" + "="*60)
        print("Experiment 3: Continual Learning (CIFAR-10)")
        print("="*60)
        result_dir = create_result_dir(dataset, "continual")
        save_config_copy(config, result_dir)
        
        # 2. replay
        config_replay = copy.deepcopy(config)
        config_replay["use_replay"] = True
        replay_dir = os.path.join(result_dir, "replay")
        os.makedirs(replay_dir, exist_ok=True)
        history_replay, summary_replay = run_continual(config_replay, replay_dir)

        # # 1. no replay
        config_no_replay = copy.deepcopy(config)
        config_no_replay["use_replay"] = False
        no_replay_dir = os.path.join(result_dir, "no_replay")
        os.makedirs(no_replay_dir, exist_ok=True)
        history_no_replay, summary_no_replay = run_continual(config_no_replay, no_replay_dir)
        
        # 3. comparison plot
        plot_continual_comparison(
            history_no_replay,
            history_replay,
            result_dir,
            experiment_name="continual_cifar10"
        )
    
    # 联邦持续学习实验
    if mode in ["fed_continual", "all"]:
        print("\n" + "="*60)
        print("Experiment 4: Federated Continual Learning (CIFAR-10)")
        print("="*60)
        result_dir = create_result_dir(dataset, "fed_continual")
        save_config_copy(config, result_dir)

        config_replay = copy.deepcopy(config)
        config_replay["use_replay"] = True
        replay_dir = os.path.join(result_dir, "replay")
        os.makedirs(replay_dir, exist_ok=True)
        history_replay, summary_replay = run_federated_continual(config_replay, replay_dir)

        config_no_replay = copy.deepcopy(config)
        config_no_replay["use_replay"] = False
        no_replay_dir = os.path.join(result_dir, "no_replay")
        os.makedirs(no_replay_dir, exist_ok=True)
        history_no_replay, summary_no_replay = run_federated_continual(config_no_replay, no_replay_dir)

        plot_continual_comparison(
            history_no_replay,
            history_replay,
            result_dir,
            experiment_name="fed_continual_cifar10"
        )

if __name__ == "__main__":
    main()