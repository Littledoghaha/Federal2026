# 联邦学习项目 (Federated Learning)

## 📋 项目概述

这是一个**联邦学习（Federated Learning）** 项目，使用 **fedavg 算法**进行分布式模型训练。项目支持 CIFAR-10 和 SVHN 两个数据集，实现了集中式学习、联邦学习和持续学习三种训练模式。

### 核心特性

- ✅ fedavg 联邦平均算法完整实现
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
│   ├── fedavg.py                      # 【fedavg 核心算法】
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
│   │   ├── fedavg_cifar10.py          # fedavg + CIFAR-10 实验
│   │   ├── centralized_cifar10.py     # 集中式训练基准实验
│   │   └── continual_cifar10.py       # 持续学习实验
│   │
│   ├── utils/                         # 【工具函数模块】
│   │   ├── __init__.py
│   │   ├── config.py                  # 配置文件读写
│   │   └── results_standard.py           # 结果保存与可视化
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

### 2️⃣ **src/fedavg.py** - fedavg 联邦学习核心算法

**功能**：实现 fedavg 算法的完整流程

| 函数                                          | 参数                                                         | 返回值                   | 功能描述                                       |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------ | ---------------------------------------------- |
| `average_state_dicts(state_dicts, weights)` | 模型状态字典列表、权重列表                                   | 平均后的状态字典         | 对多个客户端模型按权重进行平均                 |
| `fedavg_round(...)`                         | 全局模型、客户端DataLoader、设备、本地轮数、学习率等         | 更新后的全局模型         | 执行一轮联邦学习：本地训练→模型收集→加权平均 |
| `run_fedavg(...)`                           | 全局模型、客户端DataLoader、测试集DataLoader、设备、总轮数等 | (最终模型, 训练历史字典) | 运行多轮联邦学习完整过程                       |

**fedavg 算法流程**：

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

### 🔟 **src/utils/results_standard.py** - 结果管理与可视化

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

**功能**：fedavg 算法在 CIFAR-10 上的基准实验

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

**功能**：模拟"防灾忘"（Catastrophic Forgetting）场景，评估模型在学习新任务时是否遗忘旧任务

**核心函数**：

| 函数                         | 功能描述                         |
| ---------------------------- | -------------------------------- |
| `split_cifar10_by_class()` | 按类别分割 CIFAR-10 为两个任务   |
| `run_continual_learning()` | 执行完整的持续学习流程和对比实验 |

**实验设计**：

**Phase 1: 学习任务1**

- 任务：CIFAR-10 前 5 个类别（0-4）：飞机、汽车、鸟、猫、鹿
- 目标：在任务1上训练模型至收敛
- 输出：Task 1 准确率

**Phase 2: 学习任务2 + 经验重放（防遗忘）**

- 任务：CIFAR-10 后 5 个类别（5-9）：狗、青蛙、马、船、卡车
- 策略：混合新数据（Task 2）+ 旧数据样本（20% Task 1 数据用于重放）
- 评估指标：
  - Task 1 准确率（是否遗忘？）
  - Task 2 准确率（新任务学习效果）
- 输出：两个任务的性能曲线

**关键观察**：

```
  准确率
    │                    Task 2 开始学习 ▼
    │  Task 1 单独学习   ┌─────────────────
  0.9 ├─────────────────┤  ╱ 无经验重放（遗忘）
    │                  ╲╱
  0.8 │                 ╲
    │                  ╲┌────  有经验重放（缓解遗忘）
    │                   └─────────
  0.7 │
    └─────────────────────────────────>
      Round   Phase 1 完成   Phase 2: Round 1-N
```

**实验输出** (`continual_history.json`):

```json
{
  "phase": ["Task1", "Task1", ..., "Task2", "Task2", ...],
  "round": [1, 2, ..., 1, 2, ...],
  "task1_test_acc": [0.2, 0.35, ..., 0.30, 0.28, ...],
  "task2_test_acc": [null, null, ..., 0.15, 0.25, ...]
}
```

**如何运行持续学习实验**：

```bash
# 从项目根目录
python src/main.py continual
```

**性能对标**：

- 优秀：Task 1 在 Phase 2 准确率 > 85% 原性能
- 良好：Task 1 在 Phase 2 准确率 > 75% 原性能
- 一般：Task 1 在 Phase 2 准确率 > 60% 原性能
- 灾难遗忘：Task 1 在 Phase 2 准确率 < 50% 原性能

---

## ⚙️ 配置文件（src/config.json）

```json
{
  "dataset": "cifar10",
  "num_clients": 3,
  "num_rounds": 3,
  "local_epochs": 1,
  "train_batch_size": 64,
  "test_batch_size": 128,
  "lr": 0.001,
  "weight_decay": 0.0,
  "seed": 42,
  "device": "cpu"
}
```

**参数说明**：

| 参数                 | 说明                          | 默认值  | 推荐范围    |
| -------------------- | ----------------------------- | ------- | ----------- |
| `dataset`          | 数据集选择（cifar10/svhn）    | cifar10 | cifar10     |
| `num_clients`      | 联邦学习中的客户端数量        | 3       | 3-20        |
| `num_rounds`       | 联邦学习/集中式学习的轮数     | 3       | 3-50        |
| `local_epochs`     | 每轮客户端本地训练的 epoch 数 | 1       | 1-5         |
| `train_batch_size` | 训练批大小                    | 64      | 32/64/128   |
| `test_batch_size`  | 测试批大小                    | 128     | 128/256     |
| `lr`               | 学习率（Adam 优化器）         | 0.001   | 0.0001-0.01 |
| `weight_decay`     | L2 正则化系数                 | 0.0     | 0.0-0.0001  |
| `seed`             | 随机种子（确保可复现性）      | 42      | 任意整数    |
| `device`           | 计算设备（cuda/cpu）          | cpu     | cuda/cpu    |

**配置示例**：

| 场景                 | 推荐配置                                                                        |
| -------------------- | ------------------------------------------------------------------------------- |
| **快速测试**   | `num_rounds: 3, local_epochs: 1, num_clients: 2, device: "cpu"`               |
| **小规模实验** | `num_rounds: 10, local_epochs: 2, num_clients: 5, device: "cpu/cuda"`         |
| **完整实验**   | `num_rounds: 20, local_epochs: 5, num_clients: 10, device: "cuda"`            |
| **高精度测试** | `num_rounds: 50, local_epochs: 3, num_clients: 5, lr: 0.0005, device: "cuda"` |

---

## 🚀 快速开始

### 1. 环境设置

```bash
# 创建虚拟环境（推荐）
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

从项目根目录执行以下命令：

**运行所有实验**（集中式 + 联邦 + 持续学习）：

```bash
python src/main.py all
```

**运行单个实验**：

```bash
# 仅联邦学习
python src/main.py federated

# 仅集中式学习
python src/main.py centralized

# 仅持续学习（防灾忘测试）
python src/main.py continual

# 如果不指定模式，默认运行所有实验
python src/main.py
```

### 3. 查看结果

结果保存在 `results/` 目录下，每个实验生成独立的时间戳目录：

```
results/
├── cifar10_centralized_20260407_120530/
│   ├── config.json
│   ├── centralized_model_final.pth
│   ├── history.json
│   ├── history.csv
│   ├── centralized_test_loss.png
│   └── centralized_test_acc.png
├── cifar10_federated_20260407_120530/
│   ├── config.json
│   ├── global_model_final.pth
│   ├── history.json
│   ├── history.csv
│   ├── fedavg_test_loss.png
│   └── fedavg_test_acc.png
└── cifar10_continual_20260407_120530/
    ├── config.json
    ├── continual_model.pth
    ├── continual_history.json
    ├── continual_history.csv
    └── ...
```

- **JSON 格式**: 机器可读，便于编程处理
- **CSV 格式**: 可在 Excel/表格工具中查看
- **PNG 图表**: 可视化训练曲线（损失和准确率）
- **模型文件**: PyTorch `.pth` 格式，可用于推理或微调

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

#### [src/experiments/fedavg_cifar10.py](src/experiments/fedavg_cifar10.py) - fedavg + CIFAR-10 实验

| 函数                                     | 功能描述                                                                       |
| ---------------------------------------- | ------------------------------------------------------------------------------ |
| `run(config, save_dir)`                | 执行 fedavg 在 CIFAR-10 上的完整实验：准备数据→初始化模型→运行训练→保存结果 |
| `load_config(config_path)`             | 加载配置文件（JSON 格式）                                                      |
| `create_result_dir(experiment_name)`   | 创建带时间戳的结果目录                                                         |
| `save_config_copy(config, result_dir)` | 保存实验配置备份                                                               |

---

#### [src/experiments/centralized_cifar10.py](src/experiments/centralized_cifar10.py) - 集中式训练基准

集中式训练对比实验，用于比较 fedavg 与传统集中式学习的性能差异。

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

#### [src/utils/results_standard.py](src/utils/results_standard.py) - 结果处理与可视化

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
加载 config.json → 创建结果目录 → 保存配置备份 → 运行 fedavg 实验 → 保存模型和结果
```

---

## 💡 使用示例

### 示例 1：修改配置并运行联邦学习

编辑 `src/config.json`，设置更多客户端和轮数：

```json
{
  "dataset": "cifar10",
  "num_clients": 10,
  "num_rounds": 20,
  "local_epochs": 3,
  "train_batch_size": 32,
  "test_batch_size": 128,
  "lr": 0.0005,
  "weight_decay": 0.0001,
  "seed": 123,
  "device": "cuda"
}
```

运行实验：

```bash
python src/main.py federated
```

### 示例 2：快速测试模式

用于快速验证环境配置（仅运行 3 轮，1 epoch）：

编辑 `src/config.json`：

```json
{
  "dataset": "cifar10",
  "num_clients": 2,
  "num_rounds": 3,
  "local_epochs": 1,
  "train_batch_size": 64,
  "test_batch_size": 128,
  "lr": 0.001,
  "weight_decay": 0.0,
  "seed": 42,
  "device": "cpu"
}
```

运行所有实验：

```bash
python src/main.py all
```

预计耗时：CPU 约 2-5 分钟，GPU 约 30-60 秒

### 示例 3：加载已训练的模型用于推理

```python
import torch
from src.model import SimpleCNN

# 初始化模型
model = SimpleCNN(num_classes=10)

# 加载权重
state_dict = torch.load("results/cifar10_federated_YYYYMMDD_HHMMSS/global_model_final.pth")
model.load_state_dict(state_dict)
model.eval()

# 推理（不更新参数）
with torch.no_grad():
    output = model(input_tensor)  # input_tensor: [B, 3, 32, 32]
    predictions = torch.argmax(output, dim=1)
```

### 示例 4：查看客户端数据分布

```python
from src.datasets.data_split import iid_split_indices, print_client_summary
from src.datasets.cifar_svhn import load_cifar10

# 加载数据
train_set, val_set, test_set = load_cifar10(root="data")

# 进行 IID 分割
client_indices = iid_split_indices(train_set, num_clients=5, seed=42)

# 打印分布信息
print_client_summary(train_set, client_indices, "CIFAR-10")
```

输出示例：

```
========== CIFAR-10 Client Data Summary ==========
Total samples: 45000 (90% of training set)

Client 0: 9000 samples | Classes: [0:900, 1:900, ..., 9:900]
Client 1: 9000 samples | Classes: [0:900, 1:900, ..., 9:900]
Client 2: 9000 samples | Classes: [0:900, 1:900, ..., 9:900]
Client 3: 9000 samples | Classes: [0:900, 1:900, ..., 9:900]
Client 4: 9000 samples | Classes: [0:900, 1:900, ..., 9:900]
```

### 示例 5：比较三种训练模式

```bash
# 依次运行所有实验
python src/main.py all

# 在 results/ 目录对比结果
ls results/

# 查看每个模式的最终准确率
cat results/cifar10_centralized_*/history.csv | tail -1
cat results/cifar10_federated_*/history.csv | tail -1
cat results/cifar10_continual_*/continual_history.csv | tail -1
```

期望对比：

| 模式   | 最终准确率 | 特点               |
| ------ | ---------- | ------------------ |
| 集中式 | ~60-70%    | 单机训练，精度最高 |
| 联邦   | ~55-65%    | 多客户端，通信开销 |
| 持续   | ~50-60%    | 防遗忘，模拟多任务 |

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
"num_clients": 10  # 从默认的 3 改为 10
```

然后运行：

```bash
python src/main.py federated
```

### Q2: 如何使用 GPU 加速训练？

**A**: 编辑 `src/config.json`，修改 `device` 参数：

```json
"device": "cuda"  # 改为 cuda（需要已安装 CUDA 和 GPU 驱动）
```

检查 GPU 可用性：

```python
import torch
print(torch.cuda.is_available())  # True 表示 GPU 可用
print(torch.cuda.get_device_name(0))  # 获取 GPU 名称
```

### Q3: 如何设置自己的数据分割策略（非 IID）？

**A**: 修改 [src/datasets/data_split.py](src/datasets/data_split.py)，实现自定义分割函数：

```python
def non_iid_split_indices(dataset, num_clients, dirichlet_alpha=0.5, seed=42):
    """
    非 IID 分割：按 Dirichlet 分布模拟真实的非均匀数据分布
    """
    import numpy as np
    np.random.seed(seed)
  
    labels = np.array(dataset.targets if hasattr(dataset, 'targets') else 
                      dataset.dataset.targets)
    num_classes = len(np.unique(labels))
  
    # 使用 Dirichlet 分布生成权重
    proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_classes),
                                      num_clients)
  
    # 根据权重分配数据...
    return client_indices
```

### Q4: 模型训练速度慢？如何加快？

**A**: 尝试以下优化：

1. **减少训练规模**（快速测试）：

   ```json
   "num_rounds": 3,
   "local_epochs": 1,
   "num_clients": 2
   ```
2. **提大批大小**（需要更多内存）：

   ```json
   "train_batch_size": 128,
   "test_batch_size": 256
   ```
3. **使用 GPU**：

   ```json
   "device": "cuda"
   ```
4. **减少客户端数量**：

   ```json
   "num_clients": 5
   ```

### Q5: 如何在自己的数据集上运行？

**A**: 在 [src/datasets/data_loader.py](src/datasets/data_loader.py) 中添加新数据集加载函数：

```python
def load_my_dataset(root, val_ratio=0.1, seed=42):
    """加载自定义数据集"""
    # 加载数据...
    return train_set, val_set, test_set
```

修改 [src/main.py](src/main.py) 中的数据加载调用。

### Q6: 训练中断如何恢复？

**A**: 当前版本不支持断点续训。建议：

1. 保存最后一个检查点（`.pth` 文件）
2. 修改代码支持加载检查点
3. 或重新运行完整训练

### Q7: 如何理解持续学习实验的结果？

**A**: 观察 `continual_history.csv` 中的数据：

```csv
phase,round,task1_test_acc,task2_test_acc
Task1,1,0.15,
Task1,2,0.30,
Task1,3,0.45,
Task2,1,0.45,0.10       ← Task1 准确率从 0.45 下降
Task2,2,0.42,0.20       ← 继续下降（灾难遗忘现象）
Task2,3,0.38,0.35
```

- 如果 Task1 准确率在 Phase 2 **显著下降**（>20%），说明模型发生了**灾难遗忘**
- 经验重放（replay）策略能显著缓解这一问题
- 目标是在学习新任务时最小化旧任务的遗忘

### Q8: 模型在测试集上准确率很低？

**A**: 可能原因和解决方案：

| 原因             | 解决方案                                     |
| ---------------- | -------------------------------------------- |
| 训练轮数太少     | 增加 `num_rounds` 和 `local_epochs`      |
| 学习率太大或太小 | 调整 `lr`：尝试 0.0001 - 0.01 范围         |
| 正则化太强       | 减少 `weight_decay`                        |
| 客户端样本过少   | 增加 `train_batch_size` 或 `num_clients` |
| 模型欠拟合       | 确保使用足够复杂的模型（已内置 SimpleCNN）   |

### Q9: 如何导出 ONNX 模型用于部署？

**A**: 转换为 ONNX 格式：

```python
import torch
from src.model import SimpleCNN

model = SimpleCNN(num_classes=10)
model.load_state_dict(torch.load("results/.../global_model_final.pth"))

# 导出为 ONNX
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "model.onnx", 
                  input_names=['image'], output_names=['logits'])
```

### Q10: 如何在多 GPU 上并行训练？

**A**: 当前实现为单 GPU 设计。若要多 GPU 支持，需修改 [src/fedavg.py](src/fedavg.py) 中的模型分发逻辑。建议使用 `torch.nn.DataParallel`：

```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

---

## 性能指标与预期

### 典型性能表现

在 CIFAR-10 数据集上的预期准确率（约 10-20 轮训练后）：

| 模式   | 客户端数 | 轮数 | 预期准确率 | 耗时（单核 CPU） |
| ------ | -------- | ---- | ---------- | ---------------- |
| 集中式 | -        | 10   | 60-70%     | ~15 分钟         |
| 联邦   | 5        | 10   | 55-65%     | ~20 分钟         |
| 持续   | 3        | 3+3  | 50-60%     | ~10 分钟         |

**注**：

- 实际性能取决于硬件、数据集、超参数等多个因素
- GPU 加速可将耗时降低 10-20 倍
- 首次运行会下载 CIFAR-10 数据集，耗时额外 5-10 分钟

### 中间输出示例

训练过程中会输出类似信息：

```
Experiment 2: Federated Learning
============================================================

--- Round 1/10 ---
Client 0: Training... Loss=2.3042, Acc=0.1024
Client 1: Training... Loss=2.2964, Acc=0.1028
Client 2: Training... Loss=2.3001, Acc=0.1020
Client 3: Training... Loss=2.2987, Acc=0.1015
Client 4: Training... Loss=2.2956, Acc=0.1025

Aggregating models...
Test Loss: 2.2988, Test Acc: 0.1022

--- Round 2/10 ---
...

✓ Results saved to: results/cifar10_federated_20260407_120530/
```

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
│ ├─ train_local()   │ │  │ ├─ fedavg    ││
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

## 📋 更新日志

### v1.0 (2026-04-07) - 首个完整版本

**功能完成**：

- ✅ FedAvg 联邦学习算法完整实现
- ✅ 集中式学习基准对比
- ✅ 持续学习（防灾忘）实验框架
- ✅ 支持 CIFAR-10 数据集
- ✅ IID 数据分割策略
- ✅ 配置驱动的灵活性
- ✅ 完整的结果导出（JSON/CSV/PNG）

**已知限制**：

- 单 GPU/CPU 支持（无多 GPU 并行）
- 仅支持 CIFAR-10（SVHN 支持在准备中）
- 无断点续训功能

**计划中的特性**：

- 非 IID 数据分割（Dirichlet 分布）
- 多 GPU 分布式训练
- 更多数据集支持（ImageNet、MNIST 等）
- 模型压缩与优化
- Web UI 监控面板
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

## 📚 参考资源

### 学术论文

1. **FedAvg Algorithm**
   - McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." *AISTATS 2017*
   - arXiv: https://arxiv.org/abs/1602.05629

2. **Continual Learning & Catastrophic Forgetting**
   - Kirkpatrick, James, et al. "Overcoming catastrophic forgetting in neural networks." *PNAS 2017*
   - https://arxiv.org/abs/1612.00796

3. **Federated Learning Survey**
   - Yang, Qiang, et al. "Federated machine learning: Concept and applications." *ACM TIST 2019*

### 官方文档

- PyTorch: https://pytorch.org/docs/
- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

### 相关项目

- TensorFlow Federated: https://www.tensorflow.org/federated
- LEAF Dataset: https://leaf.cmu.edu/
- FedProx: https://arxiv.org/abs/1812.06127

---

## 许可

本项目采用 MIT 许可证。有关详细信息，请参阅 LICENSE 文件。

---

**项目维护者**: Federal Learning Team  
**最后更新**: 2026年4月7日  
**当前版本**: 1.0
```
