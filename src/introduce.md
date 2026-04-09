# src 目录结构与功能说明

## 目录结构

```
src/
├── main.py                    # 项目主入口
├── model.py                   # 模型定义
├── fedavg.py                  # 联邦平均算法
├── train_eval.py              # 训练与评估
├── client_dataloader.py        # DataLoader 工具
├── config.json                # 配置文件
├── datasets/                  # 数据集处理
│   ├── __init__.py
│   ├── cifar_svhn.py          # 数据集加载
│   ├── data_loader.py         # 数据流程编排
│   └── data_split.py          # 数据集分割
├── experiments/               # 实验脚本
│   ├── __init__.py
│   ├── centralized_cifar10.py # 集中式学习脚本
│   ├── fedavg_cifar10.py      # 联邦学习脚本
│   └── continual_cifar10.py   # 持续学习脚本
└── utils/                     # 工具模块
    ├── config.py              # 配置管理
    ├── results_standard.py    # 标准学习结果处理
    └── results_continual.py   # 持续学习结果处理
```

## 核心模块功能

### 顶层文件

| 文件                           | 功能                                                              |
| ------------------------------ | ----------------------------------------------------------------- |
| **main.py**              | 项目总入口，支持运行三种实验模式：centralized/federated/continual |
| **model.py**             | 模型定义模块，包含 SimpleCNN 和 DecomposedDense 等网络层          |
| **fedavg.py**            | 实现 FedAvg 联邦学习算法，包括全局参数加权平均                    |
| **train_eval.py**        | 包含训练循环、本地训练、模型评估等基础函数                        |
| **client_dataloader.py** | DataLoader 工厂函数，为训练/测试/多客户端构造数据加载器           |

### datasets/ 数据处理模块

| 文件                     | 功能                                                      |
| ------------------------ | --------------------------------------------------------- |
| **cifar_svhn.py**  | 加载原始 CIFAR-10 和 SVHN 数据集，支持划分 train/val/test |
| **data_split.py**  | 实现数据集分割算法，支持 IID 平均切分给多个客户端         |
| **data_loader.py** | 统一数据准备流程，根据实验类型返回对应的 DataLoader       |

### experiments/ 实验脚本

| 文件                             | 功能                                                       |
| -------------------------------- | ---------------------------------------------------------- |
| **centralized_cifar10.py** | 集中式学习基线实验，在完整 CIFAR-10 上训练单个模型         |
| **fedavg_cifar10.py**      | 联邦学习实验，多客户端本地训练 + 全局聚合                  |
| **continual_cifar10.py**   | 持续学习实验，分两个任务阶段，观察灾难性遗忘，支持经验重放 |

### utils/ 工具模块

| 文件                           | 功能                                        |
| ------------------------------ | ------------------------------------------- |
| **config.py**            | 配置加载、结果目录创建、配置文件保存        |
| **results_standard.py**  | 保存和分析标准学习实验结果（CSV/JSON/图表） |
| **results_continual.py** | 持续学习结果处理，计算遗忘指标并生成可视化  |

## 使用示例

```bash
# 运行所有实验
python main.py all

# 运行特定实验
python main.py centralized   # 集中式学习
python main.py federated     # 联邦学习
python main.py continual     # 持续学习
```
