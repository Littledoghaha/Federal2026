# 联邦学习项目 (Federated Learning)

## 📋 项目概述

这是一个**联邦学习（Federated Learning）** 项目，使用 **FedAvg 算法**进行分布式模型训练。项目支持 CIFAR-10 和 SVHN 两个数据集，实现了集中式学习、联邦学习和持续学习三种训练模式。

### 核心特性

- ✅ FedAvg 联邦平均算法完整实现
- ✅ 支持 CIFAR-10、SVHN 数据集
- ✅ IID 数据分割（客户端间均匀分布）
- ✅ 多客户端分布式训练
- ✅ 加权模型聚合
- ✅ 集中式学习基准实验
- ✅ 持续学习场景支持
- ✅ 训练历史自动保存（JSON/CSV）
- ✅ 可视化曲线生成

---

## 📁 完整项目结构

```
federal/
├── README.md                          # 项目文档（本文件）
├── requirements.txt                   # 依赖项列表
├── config.json                        # 默认配置文件
│
├── src/                               # 【主源代码目录】
│   ├── __init__.py
│   ├── main.py                        # 【项目入口脚本】- 协调三种实验模式
│   ├── config.json                    # 实验配置文件
│   ├── fedavg.py                      # 【FedAvg 核心算法】
│   ├── model.py                       # 【神经网络模型定义】
│   ├── train_eval.py                  # 【训练与评估通用函数】
│   ├── client_dataloader.py           # 【DataLoader 构建工具】
│   │
│   ├── datasets/                      # 【数据集处理模块】
│   │   ├── __init__.py
│   │   ├── cifar_svhn.py              # 数据集加载（CIFAR-10/SVHN）
│   │   ├── data_loader.py             # 统一数据准备流程
│   │   └── data_split.py              # 数据分割逻辑（IID）
│   │
│   ├── experiments/                   # 【实验模块】
│   │   ├── __init__.py
│   │   ├── fedavg_cifar10.py          # FedAvg + CIFAR-10 实验
│   │   ├── centralized_cifar10.py     # 集中式训练基准实验
│   │   └── continual_cifar10.py       # 持续学习实验
│   │
│   ├── utils/                         # 【工具函数模块】
│   │   ├── __init__.py
│   │   ├── config.py                  # 配置文件读写
│   │   └── results_utils.py           # 结果保存与可视化
│   │
│   ├── data/                          # 数据缓存目录
│   ├── fl/                            # 联邦学习相关模块（预留）
│   ├── cl/                            # 持续学习相关模块（预留）
│   └── models/                        # 预训练模型目录
│
├── data/                              # 全局数据目录
│   ├── test_32x32.mat                 # SVHN 测试集
│   ├── train_32x32.mat                # SVHN 训练集
│   └── cifar-10-batches-py/           # CIFAR-10 数据集
│       ├── batches.meta
│       ├── data_batch_{1-5}
│       ├── test_batch
│       └── readme.html
│
├── archive/                           # 存档数据
│   ├── PASCAL_VOC/                    # VOC 数据集（存档）
│   ├── VOCtest_06-Nov-2007/           # VOC 测试集
│   └── VOCtrainval_06-Nov-2007/       # VOC 训练验证集
│
├── results/                           # 【训练结果输出】
│   ├── fedavg_cifar10_YYYYMMDD_HHMMSS/    # 联邦学习结果
│   │   ├── config.json                     # 实验配置备份
│   │   ├── global_model_final.pth          # 最终全局模型权重
│   │   ├── history.json                    # 训练历史（JSON 格式）
│   │   ├── history.csv                     # 训练历史（CSV 格式）
│   │   ├── fedavg_cifar10_test_loss.png    # 测试损失曲线
│   │   └── fedavg_cifar10_test_acc.png     # 测试准确率曲线
│   │
│   ├── fedavg_cifar10_centralized_YYYYMMDD_HHMMSS/  # 集中式结果
│   └── fedavg_cifar10_continual_YYYYMMDD_HHMMSS/    # 持续学习结果
│
├── configs/                           # 配置文件目录（预留）
├── notebooks/                         # Jupyter notebooks
└── outputs/                           # 其他输出目录
```

---

## 📚 核心模块详解

### 1️⃣ **src/main.py** - 项目入口

**功能**：协调执行三种实验模式

| 函数       | 功能描述                               |
| ---------- | -------------------------------------- |
| `main()` | 主函数。根据命令行参数执行对应实验模式 |

**用法**：

```bash
# 运行所有实验
python -m src.main all

# 仅运行集中式学习
python -m src.main centralized

# 仅运行联邦学习
python -m src.main federated

# 仅运行持续学习
python -m src.main continual
```

**支持的模式**：

- `all` - 执行三种实验
- `centralized` - 集中式学习（单服务器）
- `federated` - 联邦学习（多客户端）
- `continual` - 持续学习（防灾忘测试）

---

### 2️⃣ **src/fedavg.py** - FedAvg 联邦学习核心算法

**功能**：实现 FedAvg 算法的完整流程

| 函数                                          | 参数                                                         | 返回值                   | 功能描述                                       |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------ | ---------------------------------------------- |
| `average_state_dicts(state_dicts, weights)` | 模型状态字典列表、权重列表                                   | 平均后的状态字典         | 对多个客户端模型按权重进行平均                 |
| `fedavg_round(...)`                         | 全局模型、客户端DataLoader、设备、本地轮数、学习率等         | 更新后的全局模型         | 执行一轮联邦学习：本地训练→模型收集→加权平均 |
| `run_fedavg(...)`                           | 全局模型、客户端DataLoader、测试集DataLoader、设备、总轮数等 | (最终模型, 训练历史字典) | 运行多轮联邦学习完整过程                       |

**FedAvg 算法流程**：

1. 初始化全局模型参数
2. 每一轮联邦聚合：
   - 将全局模型复制到所有客户端
   - 每个客户端使用本地数据进行多轮训练
   - 收集所有客户端的更新模型
   - 按数据量比例对模型参数进行加权平均
   - 更新全局模型
3. 在测试集上评估全局模型性能
4. 记录每轮的损失和准确率

---

### 3️⃣ **src/model.py** - 神经网络模型

**功能**：定义用于 CIFAR-10 和 SVHN 的卷积神经网络

| 类            | 方法                                        | 功能描述                                   |
| ------------- | ------------------------------------------- | ------------------------------------------ |
| `SimpleCNN` | `__init__(num_classes=10, in_channels=3)` | 初始化模型（10分类，3通道RGB输入）         |
|               | `forward(x)`                              | 前向传播，输入为 [B, 3, 32, 32] 的图像张量 |

**模型架构**：

```
输入: [B, 3, 32, 32]
  ↓
特征提取层:
  - Conv2d(3→32) + ReLU + MaxPool(32→16)
  - Conv2d(32→64) + ReLU + MaxPool(16→8)
  - Conv2d(64→128) + ReLU + AdaptiveAvgPool(→4×4)
  ↓
分类层:
  - Flatten: [B, 2048]
  - Linear(2048→256) + ReLU + Dropout(0.2)
  - Linear(256→10)
  ↓
输出: logits [B, 10]
```

---

### 4️⃣ **src/train_eval.py** - 训练与评估函数

**功能**：提供单个客户端的本地训练和全局模型评估函数

| 函数                                                                            | 参数                                     | 返回值                   | 功能描述                     |
| ------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------ | ---------------------------- |
| `train_one_epoch(model, dataloader, optimizer, criterion, device)`            | 模型、数据加载器、优化器、损失函数、设备 | (平均损失, 平均准确率)   | 训练单个 epoch               |
| `train_local(model, train_loader, device, epochs, lr, weight_decay, verbose)` | 模型、训练数据集、设备、轮数、学习率等   | (训练后的模型, 历史记录) | 客户端本地训练（多个 epoch） |
| `evaluate(model, dataloader, device)`                                         | 模型、测试数据集、设备                   | (测试损失, 测试准确率)   | 在测试集上评估模型性能       |

**本地训练流程**：

- 使用 Adam 优化器
- 交叉熵损失函数
- 记录每个 epoch 的损失和准确率

---

### 5️⃣ **src/client_dataloader.py** - DataLoader 工具

**功能**：构建各种数据加载器

| 函数                                                               | 功能描述                               |
| ------------------------------------------------------------------ | -------------------------------------- |
| `make_dataloader(dataset, batch_size, shuffle, num_workers)`     | 从数据集创建单个 DataLoader            |
| `build_train_loader(train_dataset, batch_size, num_workers)`     | 构建训练集 DataLoader（shuffle=True）  |
| `build_client_loaders(client_datasets, batch_size, num_workers)` | 为所有客户端构建 DataLoader 字典       |
| `build_test_loader(test_dataset, batch_size, num_workers)`       | 构建测试集 DataLoader（shuffle=False） |

---

### 6️⃣ **src/datasets/cifar_svhn.py** - 数据集加载

**功能**：下载和加载 CIFAR-10 及 SVHN 数据集

| 函数                                               | 功能描述                             |
| -------------------------------------------------- | ------------------------------------ |
| `get_default_transform()`                        | 返回默认的数据转换管道（转为张量）   |
| `load_cifar10(root, transform, val_ratio, seed)` | 加载 CIFAR-10，分割为 train/val/test |
| `load_svhn(root, transform, val_ratio, seed)`    | 加载 SVHN，分割为 train/val/test     |
| `load_all_datasets()`                            | 同时加载 CIFAR-10 和 SVHN            |

---

### 7️⃣ **src/datasets/data_split.py** - 数据分割

**功能**：实现 IID 数据分割策略

| 函数                                                            | 功能描述                             |
| --------------------------------------------------------------- | ------------------------------------ |
| `get_dataset_labels(dataset)`                                 | 兼容获取 CIFAR-10 和 SVHN 的标签     |
| `iid_split_indices(dataset, num_clients, seed)`               | 平均随机分割数据集索引到多个客户端   |
| `build_client_subsets(dataset, client_indices)`               | 根据索引创建每个客户端的 Subset 对象 |
| `get_label_distribution(dataset, indices)`                    | 统计某客户端的类别分布               |
| `print_client_summary(dataset, client_indices, dataset_name)` | 打印所有客户端的数据量和类别分布     |

**数据分割特性**：

- **IID 分割**：数据在客户端间均匀随机分布
- 支持多种数据集
- 保证再现性（支持 seed）

---

### 8️⃣ **src/datasets/data_loader.py** - 统一数据准备

**功能**：为不同训练场景提供统一的数据准备入口

| 函数                                                                        | 功能描述                         |
| --------------------------------------------------------------------------- | -------------------------------- |
| `prepare_centralized_dataloaders(dataset, batch_size, ...)`               | 为集中式学习准备数据加载器       |
| `prepare_federated_dataloaders(root, num_clients, train_batch_size, ...)` | 为联邦学习准备多客户端数据加载器 |
| `prepare_continual_dataloaders(...)`                                      | 为持续学习准备任务序列数据       |

**返回数据结构示例**：

```python
{
  "cifar": {
    "client_loaders": {0: DataLoader, 1: DataLoader, ...},
    "test_loader": DataLoader
  },
  "svhn": { ... }
}
```

---

### 9️⃣ **src/utils/config.py** - 配置管理

**功能**：配置文件读写和结果目录管理

| 函数                                     | 功能描述               |
| ---------------------------------------- | ---------------------- |
| `load_config(config_path)`             | 加载 JSON 配置文件     |
| `create_result_dir(experiment_name)`   | 创建带时间戳的结果目录 |
| `save_config_copy(config, result_dir)` | 将配置备份到结果目录   |

---

### 🔟 **src/utils/results_utils.py** - 结果管理与可视化

**功能**：训练历史保存和曲线绘制

| 函数                                                 | 功能描述                 |
| ---------------------------------------------------- | ------------------------ |
| `ensure_dir(path)`                                 | 确保目录存在（递归创建） |
| `save_history_json(history, save_path)`            | 保存训练历史为 JSON 格式 |
| `save_history_csv(history, save_path)`             | 保存训练历史为 CSV 格式  |
| `plot_history(history, save_dir, experiment_name)` | 绘制测试损失和准确率曲线 |

**历史记录格式**：

```python
{
  "round": [0, 1, 2, ...],
  "test_loss": [2.3, 2.1, 1.9, ...],
  "test_acc": [0.1, 0.2, 0.35, ...]
}
```

---

## 🧪 三大实验模块

### 1. **src/experiments/fedavg_cifar10.py** - 联邦学习实验

**功能**：FedAvg 算法在 CIFAR-10 上的基准实验

**核心流程**：

1. 初始化多个客户端并分配数据
2. 执行多轮联邦聚合
3. 保存最终模型和训练历史

**输出**：

- `global_model_final.pth` - 最终全局模型
- `history.json` - JSON 格式训练历史
- `history.csv` - CSV 格式训练历史
- 损失和准确率曲线图

---

### 2. **src/experiments/centralized_cifar10.py** - 集中式学习实验

**功能**：集中式学习作为基准对比

**流程**：

- 所有数据在单个服务器上训练
- 用于与联邦学习结果进行对比

---

### 3. **src/experiments/continual_cifar10.py** - 持续学习实验

**函数**：

- `split_cifar10_by_class()` - 按类别分割数据为两个任务
- `run_continual_learning()` - 执行持续学习流程

**场景**：

- **Task 1**：CIFAR-10 前 5 个类别（0-4）
- **Task 2**：CIFAR-10 后 5 个类别（5-9）

**观察指标**：

- Task 1 的遗忘程度（准确率下降）
- Task 2 的学习性能

---

## ⚙️ 配置文件（src/config.json）

```json
{
  "experiment_name": "fedavg_cifar10",
  "device": "cuda",
  "num_clients": 5,
  "num_rounds": 5,
  "local_epochs": 2,
  "train_batch_size": 64,
  "test_batch_size": 128,
  "lr": 0.001,
  "weight_decay": 0.0,
  "seed": 42
}
```

**参数说明**：

- `num_clients` - 联邦学习中的客户端数量
- `num_rounds` - 联邦聚合的轮数
- `local_epochs` - 每个客户端每轮的本地训练轮数
- `lr` - 学习率
- `weight_decay` - L2 正则化强度

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行实验

**运行所有实验**：

```bash
cd src
python main.py all
```

**运行单个实验**：

```bash
python main.py federated    # 仅联邦学习
python main.py centralized  # 仅集中式
python main.py continual    # 仅持续学习
```

### 3. 查看结果

结果保存在 `results/` 目录下，包含：

- 训练历史（JSON 和 CSV）
- 最终模型权重
- 可视化曲线

---

## 📊 数据流图

```
[CIFAR-10 / SVHN]
    ↓
[数据加载]
    ↓
[IID 分割] ← num_clients
    ↓
[客户端数据] → [联邦学习] → [集中式学习] → [持续学习]
    ↓            ↓             ↓             ↓
[本地训练]  [全局模型]    [单个模型]    [任务序列]
    ↓            ↓             ↓             ↓
[模型聚合]  [评估测试]    [评估测试]    [防灾忘评估]
    ↓            ↓             ↓             ↓
[Results/]  [Results/]    [Results/]    [Results/]
```

---

## 🔑 关键设计特点

| 特性                   | 说明                                       |
| ---------------------- | ------------------------------------------ |
| **IID 数据分割** | 确保客户端间数据分布均匀，便于研究通信效率 |
| **加权聚合**     | 按各客户端样本数加权，保证公平性           |
| **模型深拷贝**   | 客户端获得全局模型的独立副本，防止污染     |
| **灵活配置**     | 支持 JSON 配置文件，方便参数调整           |
| **完整日志**     | 记录每轮的损失、准确率等关键指标           |
| **多实验支持**   | 单个脚本可运行多种训练模式                 |

---

#### [src/model.py](src/model.py) - 神经网络模型

| 类            | 功能描述                                                      |
| ------------- | ------------------------------------------------------------- |
| `SimpleCNN` | 轻量级 CNN 模型，适用于 3×32×32 彩色图像分类任务（10 分类） |

**架构**：

```
特征提取层：
  Conv2d(3→32) + ReLU + MaxPool(2×2)      # 32×32 → 16×16
  Conv2d(32→64) + ReLU + MaxPool(2×2)     # 16×16 → 8×8
  Conv2d(64→128) + ReLU + AdaptiveAvgPool # → 128×4×4

分类层：
  Flatten → Linear(2048→256) + ReLU + Dropout(0.2) → Linear(256→10)
```

---

#### [src/train_eval.py](src/train_eval.py) - 训练与评估

| 函数                                                                            | 功能描述                                 |
| ------------------------------------------------------------------------------- | ---------------------------------------- |
| `train_one_epoch(model, dataloader, optimizer, criterion, device)`            | 训练一个 epoch，返回平均损失和准确率     |
| `train_local(model, train_loader, device, epochs, lr, weight_decay, verbose)` | 客户端本地训练多个 epoch，记录训练历史   |
| `evaluate(model, test_loader, device)`                                        | 在测试集上评估模型，返回测试损失和准确率 |

---

### 📊 数据处理模块

#### [src/federated_data.py](src/federated_data.py) - 联邦数据流程管理

| 函数                                                                        | 功能描述                                                      |
| --------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `prepare_federated_dataloaders(root, num_clients, train_batch_size, ...)` | 联邦数据准备的完整流程：加载→分割→创建子集→构建 DataLoader |

**数据流**：

```
原始数据集 → IID 分割 → 客户端子集 → DataLoader → 联邦训练
```

---

#### [src/datasets/cifar_svhn.py](src/datasets/cifar_svhn.py) - 数据集加载

| 函数                                   | 功能描述                             |
| -------------------------------------- | ------------------------------------ |
| `get_default_transform()`            | 返回默认的数据预处理（ToTensor）     |
| `load_cifar10(root, transform)`      | 下载并加载 CIFAR-10（训练+测试集）   |
| `load_svhn(root, transform)`         | 下载并加载 SVHN（训练+测试集）       |
| `load_all_datasets(root, transform)` | 同时加载 CIFAR-10 和 SVHN（4个集合） |

---

#### [src/datasets/data_split.py](src/datasets/data_split.py) - 数据分割逻辑

| 函数                                                            | 功能描述                                                |
| --------------------------------------------------------------- | ------------------------------------------------------- |
| `get_dataset_labels(dataset)`                                 | 从数据集中提取标签（兼容 CIFAR-10 和 SVHN）             |
| `iid_split_indices(dataset, num_clients, seed)`               | IID 均匀分割：将数据集索引随机分给 num_clients 个客户端 |
| `build_client_subsets(dataset, client_indices)`               | 根据索引字典构建每个客户端的数据子集                    |
| `get_label_distribution(dataset, indices)`                    | 统计某个客户端的类别分布                                |
| `print_client_summary(dataset, client_indices, dataset_name)` | 打印所有客户端的数据摘要和类别分布                      |

---

#### [src/client_dataloader.py](src/client_dataloader.py) - DataLoader 工具

| 函数                                                               | 功能描述                                      |
| ------------------------------------------------------------------ | --------------------------------------------- |
| `make_dataloader(dataset, batch_size, shuffle, num_workers)`     | 通用 DataLoader 创建函数                      |
| `build_client_loaders(client_datasets, batch_size, num_workers)` | 为所有客户端构建训练用 DataLoader（打乱顺序） |
| `build_test_loader(test_dataset, batch_size, num_workers)`       | 构建全局测试集 DataLoader（保持顺序）         |

---

### 🧪 实验模块

#### [src/experiments/fedavg_cifar10.py](src/experiments/fedavg_cifar10.py) - FedAvg + CIFAR-10 实验

| 函数                                     | 功能描述                                                                       |
| ---------------------------------------- | ------------------------------------------------------------------------------ |
| `run(config, save_dir)`                | 执行 FedAvg 在 CIFAR-10 上的完整实验：准备数据→初始化模型→运行训练→保存结果 |
| `load_config(config_path)`             | 加载配置文件（JSON 格式）                                                      |
| `create_result_dir(experiment_name)`   | 创建带时间戳的结果目录                                                         |
| `save_config_copy(config, result_dir)` | 保存实验配置备份                                                               |

---

#### [src/experiments/centralized_cifar10.py](src/experiments/centralized_cifar10.py) - 集中式训练基准

集中式训练对比实验，用于比较 FedAvg 与传统集中式学习的性能差异。

**主要函数**：

- `set_seed(seed)` - 设置随机种子
- `load_config(config_path)` - 加载配置
- `train_one_epoch(...)` - 单个 epoch 训练
- `evaluate(...)` - 模型评估
- 完整的集中式训练流程

---

### 🔨 工具模块

#### [src/utils/config.py](src/utils/config.py) - 配置管理

| 函数                                     | 功能描述                       |
| ---------------------------------------- | ------------------------------ |
| `load_config(config_path)`             | 加载 JSON 格式的配置文件       |
| `create_result_dir(experiment_name)`   | 创建结果目录（自动添加时间戳） |
| `save_config_copy(config, result_dir)` | 保存配置文件副本到结果目录     |

---

#### [src/utils/results_utils.py](src/utils/results_utils.py) - 结果处理与可视化

| 函数                                                 | 功能描述                             |
| ---------------------------------------------------- | ------------------------------------ |
| `ensure_dir(path)`                                 | 确保目录存在（不存在则创建）         |
| `save_history_json(history, save_path)`            | 将训练历史保存为 JSON 格式           |
| `save_history_csv(history, save_path)`             | 将训练历史保存为 CSV 格式            |
| `plot_history(history, save_dir, experiment_name)` | 绘制测试损失和准确率曲线，保存为 PNG |

**输出**：

- `*_test_loss.png` - 测试损失曲线
- `*_test_acc.png` - 测试准确率曲线

---

### 📝 入口模块

#### [src/main.py](src/main.py) - 程序入口

| 函数       | 功能描述                                               |
| ---------- | ------------------------------------------------------ |
| `main()` | 程序主入口：加载配置→创建结果目录→保存配置→运行实验 |

**执行流程**：

```
加载 config.json → 创建结果目录 → 保存配置备份 → 运行 FedAvg 实验 → 保存模型和结果
```

---

## 配置文件

### config.json 示例

```json
{
  "experiment_name": "fedavg_cifar10",
  "device": "cuda",
  "num_clients": 3,
  "num_rounds": 10,
  "local_epochs": 2,
  "train_batch_size": 64,
  "test_batch_size": 128,
  "lr": 0.001,
  "weight_decay": 0.0,
  "seed": 42
}
```

### 参数说明

| 参数                 | 说明                        | 默认值         |
| -------------------- | --------------------------- | -------------- |
| `experiment_name`  | 实验名称（用于结果目录）    | fedavg_cifar10 |
| `device`           | 计算设备（cuda/cpu）        | cuda           |
| `num_clients`      | 参与联邦学习的客户端数      | 3              |
| `num_rounds`       | 联邦学习总轮数              | 10             |
| `local_epochs`     | 每轮客户端本地训练 epoch 数 | 2              |
| `train_batch_size` | 训练批大小                  | 64             |
| `test_batch_size`  | 测试批大小                  | 128            |
| `lr`               | 学习率                      | 0.001          |
| `weight_decay`     | L2 正则化参数               | 0.0            |
| `seed`             | 随机种子                    | 42             |

---

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行实验

```bash
# 进入 src 目录
cd src

# 运行所有实验（集中式、联邦、持续学习）
python main.py all

# 或运行单个实验
python main.py federated    # 仅联邦学习
python main.py centralized  # 仅集中式学习
python main.py continual    # 仅持续学习
```

### 3. 查看结果

```bash
# 结果保存在 ../results/ 目录中
ls ../results/

# 查看训练历史
cat ../results/fedavg_cifar10_YYYYMMDD_HHMMSS/history.csv

# 训练历史包含：
# - 每一轮的测试损失和准确率
# - 模型权重（.pth）
# - 配置备份（.json）
# - 可视化曲线（.png）
```

---

## 💡 使用示例

### 示例 1：自定义配置运行联邦学习

编辑 `src/config.json`：

```json
{
  "experiment_name": "fedavg_cifar10_custom",
  "device": "cuda",
  "num_clients": 10,
  "num_rounds": 20,
  "local_epochs": 3,
  "train_batch_size": 32,
  "test_batch_size": 128,
  "lr": 0.0005,
  "weight_decay": 0.0001,
  "seed": 123
}
```

```bash
cd src
python main.py federated
```

### 示例 2：加载已训练的模型

```python
import torch
from model import SimpleCNN

# 加载模型
model = SimpleCNN(num_classes=10)
model.load_state_dict(torch.load("../results/fedavg_cifar10_TIMESTAMP/global_model_final.pth"))
model.eval()
```

### 示例 3：查看数据分布

```python
from datasets.data_split import print_client_summary, iid_split_indices
from datasets.cifar_svhn import load_cifar10

train_set, val_set, test_set = load_cifar10()
client_indices = iid_split_indices(train_set, num_clients=5)
print_client_summary(train_set, client_indices, "CIFAR-10")
```

---

## 🔧 进阶配置

### 调整超参数

| 场景                 | 推荐配置                                          |
| -------------------- | ------------------------------------------------- |
| **快速测试**   | `num_rounds=3, local_epochs=1, num_clients=2`   |
| **小规模实验** | `num_rounds=10, local_epochs=2, num_clients=5`  |
| **完整实验**   | `num_rounds=20, local_epochs=5, num_clients=10` |
| **收敛性测试** | `num_rounds=50, local_epochs=3, num_clients=3`  |

### 学习率调整

```python
# config.json
"lr": 0.001      # Adam 优化器默认学习率
"weight_decay": 0.0001  # L2 正则化
```

**建议**：

- 小数据集或快速收敛：`lr = 0.001`
- 大数据集或防过拟合：`lr = 0.0005, weight_decay = 0.0001`

---

## 📊 输出文件说明

### 结果目录结构

```
results/fedavg_cifar10_20260401_190330/
├── config.json                        # 本次实验的完整配置
├── global_model_final.pth             # 最终全局模型（PyTorch .pth 格式）
├── history.json                       # 训练历史（JSON 格式）
├── history.csv                        # 训练历史（CSV 格式）
├── fedavg_cifar10_test_loss.png       # 测试损失曲线
└── fedavg_cifar10_test_acc.png        # 测试准确率曲线
```

### history.json 格式

```json
{
  "round": [0, 1, 2, 3, ...],
  "test_loss": [2.3, 2.1, 1.9, 1.7, ...],
  "test_acc": [0.1, 0.2, 0.35, 0.45, ...]
}
```

### history.csv 格式

```csv
round,test_loss,test_acc
0,2.3,0.1
1,2.1,0.2
2,1.9,0.35
3,1.7,0.45
...
```

---

## ❓ 常见问题（FAQ）

### Q1: 如何修改客户端数量？

**A**: 编辑 `src/config.json` 中的 `num_clients` 参数：

```json
"num_clients": 10  # 默认为 3
```

### Q2: 如何使用 CPU 代替 GPU？

**A**: 编辑 `src/config.json`：

```json
"device": "cpu"  # 改为 cpu
```

### Q3: 如何制作自己的数据集分割？

**A**: 修改 `datasets/data_split.py` 中的分割策略：

```python
def non_iid_split_indices(dataset, num_clients, seed=42):
    # 实现非 IID 分割
    ...
```

### Q4: 如何加载已训练的模型用于推理？

**A**:

```python
import torch
from model import SimpleCNN

model = SimpleCNN(num_classes=10)
state_dict = torch.load("results/.../global_model_final.pth")
model.load_state_dict(state_dict)
model.eval()

# 进行推理
with torch.no_grad():
    output = model(input_tensor)
```

### Q5: 如何监控训练进度？

**A**: 训练过程会实时输出每个客户端和每一轮的日志。结果实时保存到 `results/` 。

### Q6: 持续学习实验在测试什么？

**A**: 观察任务1 的准确率在学习任务2 后是否下降（灾难遗忘现象）。

---

## 📈 性能优化建议

| 优化目标             | 方法                                                                   |
| -------------------- | ---------------------------------------------------------------------- |
| **加快训练**   | 减少 `num_rounds`, `local_epochs`；增加 `batch_size`             |
| **改善准确率** | 增加 `num_rounds`, `local_epochs`；调整 `lr` 和 `weight_decay` |
| **降低内存**   | 减少 `num_clients`, `batch_size`                                   |
| **稳定性**     | 确保 `seed` 固定；使用较小的 `lr`                                  |

---

## 🤝 项目架构图

```
┌─────────────────────────────────────────────────┐
│         main.py (项目入口)                      │
│   协调 Centralized / Federated / Continual      │
└────────────┬────────────────────────────────────┘
             │
    ┌────────┴─────────┬──────────────────┐
    │                  │                  │
┌───▼────────────────┐ │  ┌──────────────┐│
│ Centralized        │ │  │ Federated    ││
│ ├─ train_local()   │ │  │ ├─ FedAvg    ││
│ ├─ evaluate()      │ │  │ ├─ Average   ││
│ └─ 结果保存         │ │  │ └─ 结果保存  ││
└────────────────────┘ │  └──────────────┘│
                       │                  │
                       │  ┌──────────────┐│
                       │  │ Continual    ││
                       │  │ 防灾忘测试   ││
                       │  └──────────────┘│
                       └──────────────────┘
                            │
                ┌───────────┬┴──────────┐
                │           │          │
         ┌──────▼─────┐ ┌──┴┬────┐ ┌──▼────────┐
         │ 数据处理    │ │配置│日志│ │结果管理   │
         │ load()     │ │mgmt│   │ │plot()     │
         │ split()    │ │    │   │ │save()     │
         └────────────┘ └────┴───┘ └───────────┘
```

---

## 📝 参考资源

- **论文**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg 算法)
- **PyTorch 文档**: https://pytorch.org/docs/
- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html

---

## ✨ 项目特点总结

| 特点                 | 详情                                     |
| -------------------- | ---------------------------------------- |
| **模块化设计** | 每个功能独立为模块，易于扩展和调试       |
| **配置驱动**   | 使用 JSON 配置文件，无需修改代码改变参数 |
| **自动记录**   | 每个实验自动保存配置、模型和训练历史     |
| **标准化接口** | 统一的数据加载和模型接口，便于集成新算法 |
| **可视化支持** | 自动生成损失和准确率曲线                 |
| **多实验对比** | 支持集中、联邦、持续学习多种场景         |
| **易于复现**   | 固定 seed 确保实验结果可复现             |

# 运行主程序

python main.py

```

### 3. 查看结果

结果将保存到 `results/fedavg_cifar10_YYYYMMDD_HHMMSS/` 目录，包含：

- `global_model_final.pth` - 最终全局模型
- `history.json` - 完整的训练历史
- `history.csv` - CSV 格式的训练历史
- `*_test_loss.png` - 测试损失曲线图
- `*_test_acc.png` - 测试准确率曲线图
- `config.json` - 本次实验的配置备份

---

## 数据处理流程

```

┌─────────────────────────────────────────────┐
│         加载原始数据集                        │
│  (CIFAR-10: 50K训练 + 10K测试)              │
│  (SVHN: ~73K训练 + ~26K测试)                │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│      IID 数据分割 (均匀随机)                 │
│   例如：10K样本分给3个客户端                │
│   Client 0: ~3333 样本                      │
│   Client 1: ~3333 样本                      │
│   Client 2: ~3334 样本                      │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│    构建客户端数据子集（Subset）             │
│  每个客户端持有自己的数据分区               │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│     构建 DataLoader                         │
│  训练: shuffle=True, batch_size=64         │
│  测试: shuffle=False, batch_size=128       │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│    准备完成，交给 FedAvg 训练               │
└─────────────────────────────────────────────┘

```

---

## 训练流程

```

┌─────────────────────────────────────────────┐
│           Round 0: 测试初始模型              │
│        (记录基准性能)                        │
└────────────┬────────────────────────────────┘
             │
        ┌────▼─────┐
        │ Round 1-N│
        └────┬─────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│    复制全局模型给每个客户端                 │
└────────────┬────────────────────────────────┘
             │
             ▼
     ┌───────┴───────┬───────────┐
     │               │           │
     ▼               ▼           ▼
┌─────────┐    ┌─────────┐  ┌─────────┐
│Client 0 │    │Client 1 │  │Client 2 │
│本地训练 │    │本地训练 │  │本地训练 │
└────┬────┘    └────┬────┘  └────┬────┘
     │              │           │
     └──────┬───────┴───┬───────┘
            │           │
            ▼           ▼
        ┌───────────────────────┐
        │  收集所有客户端模型    │
        │   {θ_0, θ_1, θ_2}    │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  加权平均 (按样本数)  │
        │  θ_global = Σ(w_i*θ_i)│
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  更新全局模型         │
        │  测试新模型性能       │
        │  记录损失和准确率     │
        └───────────┬───────────┘
                    │
                    ▼
            [是否继续下一轮？]
                    │
         ┌──────────┴──────────┐
         │                     │
        [是]                   [否]
         │                     │
         ▼                     ▼
      下一轮                 保存结果

```

---

## 输出示例

### 训练过程日志

```

Device: cuda

CIFAR-10 client sizes:
client 0: 16668 samples
client 1: 16666 samples
client 2: 16666 samples

round 0 - test_loss: 2.3026 - test_acc: 0.1000
=== Federated Round 1/10 ===
client 0 - samples: 16668 - train_loss: 2.2891 - train_acc: 0.1215
client 1 - samples: 16666 - train_loss: 2.2865 - train_acc: 0.1289
client 2 - samples: 16666 - train_loss: 2.2843 - train_acc: 0.1344
round 1 - test_loss: 1.9765 - test_acc: 0.3890
...
round 10 - test_loss: 0.8241 - test_acc: 0.7102

```

### 生成的可视化

- **test_loss.png**: 显示测试损失随联邦轮数递减的趋势
- **test_acc.png**: 显示测试准确率随联邦轮数上升的趋势

---

## 依赖项

见 [requirements.txt](requirements.txt)

主要依赖：

- PyTorch
- torchvision
- numpy
- matplotlib
- (其他支持库)

---

## 注意事项

1. **数据 IID 假设**：当前实现假设数据在客户端间均匀分布（IID），不支持非 IID 场景
2. **客户端参与**：当前实现中所有客户端都参与每一轮（不支持客户端采样）
3. **加权平均**：按各客户端样本数进行加权（样本多的客户端权重大）
4. **计算设备**：优先使用 GPU，如不可用则自动降级到 CPU

---

## 项目状态

- ✅ FedAvg 基础实现
- ✅ CIFAR-10 实验
- 🔄 Non-IID 数据分割（开发中）
- 🔄 客户端采样（开发中）
- 🔄 差分隐私（计划中）

---

## 相关文献

- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg, AISTATS 2017)

---

## 许可

[待定]

---

**最后更新**: 2026-03-31
```
