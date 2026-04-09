"""
模型定义模块。定义了模型SimpleCNN。
这里先放一个适用于 3x32x32 彩色图像、10 分类的小型 CNN。
可用于 CIFAR-10 和 SVHN。
"""

import torch.nn as nn


# # class SimpleCNN(nn.Module):
# class YiFanCNN(nn.Module):
#     def __init__(self, num_classes=10, in_channels=3):
#         super().__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),   # 32x32 -> 16x16

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),   # 16x16 -> 8x8

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((4, 4))   # -> 128 x 4 x 4
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 4 * 4, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.2),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DecomposedDense(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        use_bias=False,
        lambda_l1=None,
        lambda_mask=None,
        shared=None,
        adaptive=None,
        from_kb=None,
        atten=None,
        mask=None,
        bias=None,
        args=None,
        **kwargs,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.args = args
        self.use_bias = use_bias

        self.sw = (
            nn.Parameter(shared.clone().detach(), requires_grad=True)
            if shared is not None
            else None
        )
        self.aw = (
            nn.Parameter(adaptive.clone().detach(), requires_grad=True)
            if adaptive is not None
            else None
        )
        self.mask = (
            nn.Parameter(mask.clone().detach(), requires_grad=True)
            if mask is not None
            else None
        )
        self.bias = (
            nn.Parameter(bias.clone().detach(), requires_grad=True)
            if bias is not None
            else None
        )

        if from_kb is not None:
            self.aw_kb = nn.Parameter(from_kb.clone().detach(), requires_grad=False)
        else:
            self.aw_kb = None

        self.atten = (
            nn.Parameter(atten.clone().detach(), requires_grad=True)
            if atten is not None
            else None
        )

        self.lambda_l1 = lambda_l1
        self.lambda_mask = lambda_mask

    def l1_pruning(self, weights, hyp):
        hard_threshold = torch.gt(torch.abs(weights), hyp).float()
        return weights * hard_threshold

    def forward(self, inputs):
        aw = self.aw if self.training else self.l1_pruning(self.aw, self.lambda_l1)
        mask = (
            self.mask if self.training else self.l1_pruning(self.mask, self.lambda_mask)
        )

        theta = self.sw * mask + aw

        if self.aw_kb is not None and self.atten is not None:
            # aw_kb: [K, out, in]
            # atten: [K]
            kb_part = torch.sum(self.aw_kb * self.atten.view(-1, 1, 1), dim=0)
            theta = theta + kb_part

        return F.linear(inputs, theta, self.bias)


class DecomposedConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        activation=None,
        use_bias=False,
        lambda_l1=None,
        lambda_mask=None,
        shared=None,
        adaptive=None,
        from_kb=None,
        atten=None,
        mask=None,
        bias=None,
        args=None,
        **kwargs,
    ):
        super().__init__()

        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.activation = activation
        self.use_bias = use_bias

        self.sw = (
            nn.Parameter(shared.clone().detach(), requires_grad=True)
            if shared is not None
            else None
        )
        self.aw = (
            nn.Parameter(adaptive.clone().detach(), requires_grad=True)
            if adaptive is not None
            else None
        )
        self.mask = (
            nn.Parameter(mask.clone().detach(), requires_grad=True)
            if mask is not None
            else None
        )
        self.bias = (
            nn.Parameter(bias.clone().detach(), requires_grad=True)
            if bias is not None
            else None
        )

        if from_kb is not None:
            self.aw_kb = nn.Parameter(from_kb.clone().detach(), requires_grad=False)
        else:
            self.aw_kb = None

        self.atten = (
            nn.Parameter(atten.clone().detach(), requires_grad=True)
            if atten is not None
            else None
        )

        self.lambda_l1 = lambda_l1
        self.lambda_mask = lambda_mask

    def l1_pruning(self, weights, hyp):
        hard_threshold = torch.gt(torch.abs(weights), hyp).float()
        return weights * hard_threshold

    def forward(self, inputs):
        aw = self.aw if self.training else self.l1_pruning(self.aw, self.lambda_l1)
        mask = (
            self.mask if self.training else self.l1_pruning(self.mask, self.lambda_mask)
        )

        theta = self.sw * mask + aw

        if self.aw_kb is not None and self.atten is not None:
            # aw_kb: [K, out, in, k, k]
            # atten: [K]
            kb_part = torch.sum(self.aw_kb * self.atten.view(-1, 1, 1, 1, 1), dim=0)
            theta = theta + kb_part

        return F.conv2d(
            inputs,
            theta,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class SimpleCNN(nn.Module):
    """
    适配版 SimpleCNN：
    输入: [B, 3, 32, 32]
    输出: [B, num_classes]

    结构对应原始 SimpleCNN:
    Conv(3->32) + ReLU + Pool
    Conv(32->64) + ReLU + Pool
    Conv(64->128) + ReLU + AdaptiveAvgPool(4x4)
    Flatten
    FC(128*4*4 -> 256) + ReLU + Dropout
    FC(256 -> num_classes)
    """

    def __init__(self, args=None, num_classes=10, in_channels=3):
        super().__init__()
        if args is None:
            from types import SimpleNamespace
            args = SimpleNamespace(
                lambda_l1=1e-4,
                lambda_mask=1e-4,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        self.args = args
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.initializer = nn.init.xavier_uniform_
        self.adaptive_factor = 3

        # 这里定义每一层的 shape，和文献里的写法对应
        # 卷积层: (in_channels, out_channels, kernel_size)
        # 全连接层: (in_features, out_features)
        self.shapes = [
            (in_channels, 32, 3),  # conv1
            (32, 64, 3),  # conv2
            (64, 128, 3),  # conv3
            (128 * 4 * 4, 256),  # fc1
        ]

        self.decomposed_variables = {
            "shared": [],
            "adaptive": {},
            "mask": {},
            "bias": {},
            "atten": {},
            "from_kb": {},
        }

        # 这里只做单任务版本，如果你后面需要多任务，我也可以再帮你扩展
        self.task_id = 0

        initial_weights = self.init_global_weights()
        self.init_decomposed_variables(initial_weights)

        # ===== 特征提取部分 =====
        self.conv1 = self.conv_decomposed(0, self.task_id, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 32x32 -> 16x16

        self.conv2 = self.conv_decomposed(1, self.task_id, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 16x16 -> 8x8

        self.conv3 = self.conv_decomposed(2, self.task_id, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((4, 4))  # -> 128 x 4 x 4

        # ===== 分类部分 =====
        self.flatten = nn.Flatten()
        self.fc1 = self.dense_decomposed(3, self.task_id)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # feature extractor
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.gap(x)

        # classifier
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    # =========================
    # 文献中的变量初始化逻辑
    # =========================
    def init_global_weights(self):
        global_weights = []
        for shape in self.shapes:
            if len(shape) == 3:
                tensor = torch.empty(shape[1], shape[0], shape[2], shape[2])
            else:
                tensor = torch.empty(shape[1], shape[0])
            self.initializer(tensor)
            global_weights.append(tensor.detach().cpu().numpy())
        return global_weights

    def init_decomposed_variables(self, initial_weights):
        self.decomposed_variables["shared"] = [
            nn.Parameter(
                torch.tensor(initial_weights[i], dtype=torch.float32),
                requires_grad=True,
            )
            for i in range(len(self.shapes))
        ]

        tid = self.task_id
        for lid in range(len(self.shapes)):
            for var_type in ["adaptive", "bias", "mask", "atten", "from_kb"]:
                self.create_variable(var_type, lid, tid)

    def create_variable(self, var_type, lid, tid=0):
        trainable = True

        if tid not in self.decomposed_variables[var_type]:
            self.decomposed_variables[var_type][tid] = {}

        if var_type == "adaptive":
            init_value = (
                self.decomposed_variables["shared"][lid].detach().cpu().numpy()
                / self.adaptive_factor
            )

        elif var_type == "from_kb":
            # 单任务先不使用 KB，这里给一个长度为1的占位，避免后面 stack 出错
            if len(self.shapes[lid]) == 3:
                shape = (
                    1,
                    self.shapes[lid][1],
                    self.shapes[lid][0],
                    self.shapes[lid][2],
                    self.shapes[lid][2],
                )
            else:
                shape = (1, self.shapes[lid][1], self.shapes[lid][0])
            init_value = np.zeros(shape, dtype=np.float32)
            trainable = False

        elif var_type == "bias":
            init_value = torch.randn(self.shapes[lid][1]).detach().cpu().numpy()

        elif var_type == "mask":
            # 注意：mask 最好和 shared 权重同形状，否则广播会有问题
            if len(self.shapes[lid]) == 3:
                shape = (
                    self.shapes[lid][1],
                    self.shapes[lid][0],
                    self.shapes[lid][2],
                    self.shapes[lid][2],
                )
            else:
                shape = (self.shapes[lid][1], self.shapes[lid][0])

            tensor = torch.empty(*shape)
            self.initializer(tensor)
            init_value = tensor.detach().cpu().numpy()

        elif var_type == "atten":
            # 对 from_kb 的每个知识块给一个权重，这里只有1个
            init_value = np.ones((1,), dtype=np.float32)

        else:
            raise ValueError(f"Unknown var_type: {var_type}")

        var = torch.tensor(init_value, requires_grad=trainable, dtype=torch.float32)
        self.decomposed_variables[var_type][tid][lid] = var

    def get_variable(self, var_type, lid, tid=0):
        if var_type == "shared":
            return self.decomposed_variables[var_type][lid]
        return self.decomposed_variables[var_type][tid][lid]

    def generate_mask(self, mask):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        return torch.sigmoid(mask)

    def conv_decomposed(self, lid, tid, stride=1, padding=0):
        return DecomposedConv(
            in_channels=self.shapes[lid][0],
            out_channels=self.shapes[lid][1],
            kernel_size=self.shapes[lid][2],
            stride=stride,
            padding=padding,
            lambda_l1=self.args.lambda_l1,
            lambda_mask=self.args.lambda_mask,
            shared=self.get_variable("shared", lid),
            adaptive=self.get_variable("adaptive", lid, tid),
            from_kb=self.get_variable("from_kb", lid, tid),
            atten=self.get_variable("atten", lid, tid),
            bias=self.get_variable("bias", lid, tid),
            use_bias=True,
            mask=self.generate_mask(self.get_variable("mask", lid, tid)),
            args=self.args,
        )

    def dense_decomposed(self, lid, tid):
        return DecomposedDense(
            in_features=self.shapes[lid][0],
            out_features=self.shapes[lid][1],
            lambda_l1=self.args.lambda_l1,
            lambda_mask=self.args.lambda_mask,
            shared=self.get_variable("shared", lid),
            adaptive=self.get_variable("adaptive", lid, tid),
            from_kb=self.get_variable("from_kb", lid, tid),
            atten=self.get_variable("atten", lid, tid),
            bias=self.get_variable("bias", lid, tid),
            use_bias=True,
            mask=self.generate_mask(self.get_variable("mask", lid, tid)),
            args=self.args,
        )
