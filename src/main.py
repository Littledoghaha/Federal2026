"""
项目总入口。运行三个实验：
1. 集中式学习（baseline）
2. 联邦学习（baseline）
3. 持续学习（观察遗忘）
"""
import sys
from experiments.fedavg_cifar10 import run as run_fedavg
# load_config, create_result_dir, save_config_copy
from experiments.centralized_cifar10 import main as run_centralized
from experiments.continual_cifar10 import run_continual_learning
from utils.config import load_config,create_result_dir,save_config_copy

def main():
    config = load_config("E:\\federal\\src\\config.json")
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if mode in ["centralized", "all"]:
        print("\n" + "="*60)
        print("Experiment 1: Centralized Learning")
        print("="*60)
        run_centralized()
    
    if mode in ["federated", "all"]:
        print("\n" + "="*60)
        print("Experiment 2: Federated Learning")
        print("="*60)
        result_dir = create_result_dir(config["experiment_name"])
        save_config_copy(config, result_dir)
        run_fedavg(config, result_dir)
    
    if mode in ["continual", "all"]:
        print("\n" + "="*60)
        print("Experiment 3: Continual Learning (CIFAR-10)")
        print("="*60)
        result_dir = create_result_dir(config["experiment_name"] + "_continual")
        save_config_copy(config, result_dir)
        run_continual_learning(config, result_dir)

if __name__ == "__main__":
    main()