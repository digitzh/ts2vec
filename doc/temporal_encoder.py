"""
BearingSSL用到的时序编码器（1D CNN），供参考。

包含两种架构：
1. TemporalEncoder (Baseline): 标准的3层1D CNN架构
2. MultiScaleWaveletTemporalEncoder: 多路径多尺度小波融合编码器

参考TPN模型架构（Baseline）：
- Conv1D: 32 filters, 24 kernel, ReLU, L2 regularization
- Dropout: 10%
- Conv1D: 64 filters, 16 kernel, ReLU, L2 regularization
- Dropout: 10%
- Conv1D: 96 filters, 8 kernel, ReLU, L2 regularization
- Dropout: 10%
- GlobalMaxPool1D
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """
    时序信号编码器，使用1D卷积神经网络。

    输入形状: (batch_size, num_channels, sequence_length)
    输出形状: 
        - 如果 return_timesteps=False: (batch_size, embedding_dim)
        - 如果 return_timesteps=True: (batch_size, embedding_dim, sequence_length)
    """

    def __init__(
        self,
        num_channels: int = 1,
        sequence_length: int = 1024,
        embedding_dim: int = 96,
        dropout: float = 0.1,
        l2_reg: float = 1e-4,
        return_timesteps: bool = False,
    ):
        """
        初始化时序编码器。

        参数:
            num_channels: 输入通道数（例如：振动=1）
            sequence_length: 序列长度（窗口大小，默认1024）
            embedding_dim: 输出嵌入维度（最后一层卷积的filters数）
            dropout: Dropout比率
            l2_reg: L2正则化系数（用于weight decay）
            return_timesteps: 是否返回每个时间步的表示（True: B x embedding_dim x L, False: B x embedding_dim）
        """
        super().__init__()
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.return_timesteps = return_timesteps

        # 第一层卷积：32 filters, kernel=24
        self.conv1 = nn.Conv1d(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=24,
            padding=11,  # 保持输出长度不变（(L-1)*s - 2*p + k = L, p=(k-1)/2）
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout)

        # 第二层卷积：64 filters, kernel=16
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=16,
            padding=7,  # 保持输出长度不变
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)

        # 第三层卷积：96 filters, kernel=8
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=96,
            kernel_size=8,
            padding=3,  # 保持输出长度不变
        )
        self.bn3 = nn.BatchNorm1d(96)
        self.dropout3 = nn.Dropout(dropout)

        # 全局最大池化（仅在 return_timesteps=False 时使用）
        if not return_timesteps:
            self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_max_pool = None

        # 计算weight decay（PyTorch通过optimizer设置L2正则化）
        self.l2_reg = l2_reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: 输入张量，形状 (batch_size, num_channels, sequence_length)

        返回:
            如果 return_timesteps=False: 嵌入向量，形状 (batch_size, embedding_dim)
            如果 return_timesteps=True: 时间步表示，形状 (batch_size, embedding_dim, sequence_length)
        """
        # Conv1: (B, C, L) -> (B, 32, L)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Conv2: (B, 32, L) -> (B, 64, L)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Conv3: (B, 64, L) -> (B, 96, L)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        if self.return_timesteps:
            # 返回每个时间步的表示: (B, embedding_dim, L)
            return x
        else:
            # 全局最大池化: (B, 96, L) -> (B, 96, 1)
            x = self.global_max_pool(x)

            # 展平: (B, 96, 1) -> (B, 96)
            x = x.squeeze(-1)

            return x

    def get_weight_decay_groups(self) -> list[dict]:
        """
        返回用于weight decay的参数组（用于L2正则化）。

        返回:
            参数组列表，包含需要L2正则化的参数
        """
        return [
            {
                "params": [p for n, p in self.named_parameters() if "weight" in n],
                "weight_decay": self.l2_reg,
            },
            {
                "params": [p for n, p in self.named_parameters() if "bias" in n],
                "weight_decay": 0.0,
            },
        ]


class h_sigmoid(nn.Module):
    """
    硬sigmoid激活函数，用于SE Block。
    
    h_sigmoid(x) = ReLU6(x + 3) / 6
    输出范围: [0, 1]
    """
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEBlock1D(nn.Module):
    """
    1D Squeeze-and-Excitation Block，用于通道注意力机制。
    
    通过全局平均池化和门控机制自适应地学习各通道特征的重要性权重。
    参考: doc/SEBlock.md (适配为1D版本)
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        """
        初始化SE Block。
        
        参数:
            channels: 输入通道数
            reduction: 降维比例（默认4，即中间层维度为 channels // reduction）
        """
        super().__init__()
        self.channels = channels
        reduced_channels = max(1, channels // reduction)  # 确保至少为1
        
        # Squeeze: 全局平均池化 + 全连接层降维
        # Excitation: 全连接层升维 + 激活函数
        self.dense = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            h_sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入张量，形状 (batch_size, channels, length)
        
        返回:
            加权后的特征，形状 (batch_size, channels, length)
        """
        batch_size, channels, length = x.size()
        
        # Squeeze: 全局平均池化 (B, C, L) -> (B, C)
        squeezed = F.adaptive_avg_pool1d(x, 1).view(batch_size, channels)
        
        # Excitation: 学习通道权重 (B, C) -> (B, C)
        weights = self.dense(squeezed)  # (B, C)
        weights = weights.view(batch_size, channels, 1)  # (B, C, 1)
        
        # 应用权重: 通道级乘法
        return x * weights


class MultiScaleWaveletBranch(nn.Module):
    """
    多尺度小波分析分支。
    
    使用不同尺度的卷积核模拟不同时间-频率分辨率的小波分析。
    每个分支对应一个尺度，用于捕获瞬态冲击（高频）与调制谐波（低频）等特征。
    """
    
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        out_channels: int = 32,
        dropout: float = 0.1,
    ):
        """
        初始化多尺度小波分支。
        
        参数:
            in_channels: 输入通道数
            kernel_size: 卷积核大小（决定尺度）
            out_channels: 输出通道数（轻量化设计，默认32）
            dropout: Dropout比率
        """
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2  # 保持输出长度不变
        
        # 轻量化1D卷积编码：单层卷积
        # 使用可学习的卷积核模拟小波滤波器
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入张量，形状 (batch_size, in_channels, length)
        
        返回:
            特征图，形状 (batch_size, out_channels, length)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class MultiScaleWaveletTemporalEncoder(nn.Module):
    """
    多路径多尺度融合时序编码器。
    
    架构设计：
    1. 多路径多尺度小波分析分支：并行的多个分支，每个分支对应不同尺度
    2. 轻量化1D卷积编码：每个分支经轻量化卷积生成尺度特异性特征
    3. SE Block融合：通过通道注意力机制自适应地学习各尺度特征的重要性权重
    4. 统一嵌入表示：生成对故障敏感且鲁棒的特征嵌入
    
    输入形状: (batch_size, num_channels, sequence_length)
    输出形状: 
        - 如果 return_timesteps=False: (batch_size, embedding_dim)
        - 如果 return_timesteps=True: (batch_size, embedding_dim, sequence_length)
    """
    
    def __init__(
        self,
        num_channels: int = 1,
        sequence_length: int = 1024,
        embedding_dim: int = 96,
        dropout: float = 0.1,
        l2_reg: float = 1e-4,
        num_scales: int = 4,
        scale_kernels: list[int] | None = None,
        branch_channels: int = 32,
        se_reduction: int = 4,
        return_timesteps: bool = False,
    ):
        """
        初始化多尺度小波时序编码器。
        
        参数:
            num_channels: 输入通道数（例如：振动=1）
            sequence_length: 序列长度（窗口大小，默认1024）
            embedding_dim: 输出嵌入维度（默认96，与baseline保持一致）
            dropout: Dropout比率
            l2_reg: L2正则化系数（用于weight decay）
            num_scales: 多尺度分支数量（默认4）
            scale_kernels: 各分支的卷积核大小列表（默认None，自动生成）
            branch_channels: 每个分支的输出通道数（默认32，轻量化设计）
            se_reduction: SE Block的降维比例（默认4）
            return_timesteps: 是否返回每个时间步的表示（True: B x embedding_dim x L, False: B x embedding_dim）
        """
        super().__init__()
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.num_scales = num_scales
        self.branch_channels = branch_channels
        self.return_timesteps = return_timesteps
        
        # 如果没有指定卷积核大小，使用默认的多尺度配置
        # 从小到大对应：高频（瞬态冲击）-> 低频（调制谐波）
        if scale_kernels is None:
            # 默认使用4个尺度：[8, 16, 32, 64]
            # 对应不同的时间-频率分辨率
            scale_kernels = [8, 16, 32, 64]
        
        if len(scale_kernels) != num_scales:
            raise ValueError(
                f"scale_kernels长度({len(scale_kernels)})必须等于num_scales({num_scales})"
            )
        
        self.scale_kernels = scale_kernels
        
        # 创建多个并行的多尺度小波分支
        self.branches = nn.ModuleList([
            MultiScaleWaveletBranch(
                in_channels=num_channels,
                kernel_size=kernel_size,
                out_channels=branch_channels,
                dropout=dropout,
            )
            for kernel_size in scale_kernels
        ])
        
        # 拼接所有分支的特征
        total_channels = num_scales * branch_channels  # 例如：4 * 32 = 128
        
        # SE Block：通道注意力机制，自适应融合各尺度特征
        self.se_block = SEBlock1D(
            channels=total_channels,
            reduction=se_reduction,
        )
        
        # 投影层：将融合后的特征投影到目标嵌入维度
        self.projection = nn.Sequential(
            nn.Conv1d(
                in_channels=total_channels,
                out_channels=embedding_dim,
                kernel_size=1,  # 1x1卷积，只改变通道数
                padding=0,
            ),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # 全局最大池化（仅在 return_timesteps=False 时使用）
        if not return_timesteps:
            self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_max_pool = None
        
        # L2正则化系数
        self.l2_reg = l2_reg
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入张量，形状 (batch_size, num_channels, sequence_length)
        
        返回:
            如果 return_timesteps=False: 嵌入向量，形状 (batch_size, embedding_dim)
            如果 return_timesteps=True: 时间步表示，形状 (batch_size, embedding_dim, sequence_length)
        """
        # 多路径多尺度小波分析
        branch_outputs = []
        for branch in self.branches:
            branch_feat = branch(x)  # (B, branch_channels, L)
            branch_outputs.append(branch_feat)
        
        # 拼接所有分支的特征: (B, num_scales * branch_channels, L)
        fused = torch.cat(branch_outputs, dim=1)
        
        # SE Block：通道注意力融合，自适应加权各尺度特征
        fused = self.se_block(fused)  # (B, total_channels, L)
        
        # 投影到目标嵌入维度: (B, embedding_dim, L)
        x = self.projection(fused)
        
        if self.return_timesteps:
            # 返回每个时间步的表示: (B, embedding_dim, L)
            return x
        else:
            # 全局最大池化: (B, embedding_dim, L) -> (B, embedding_dim, 1)
            x = self.global_max_pool(x)
            
            # 展平: (B, embedding_dim, 1) -> (B, embedding_dim)
            x = x.squeeze(-1)
            
            return x
    
    def get_weight_decay_groups(self) -> list[dict]:
        """
        返回用于weight decay的参数组（用于L2正则化）。
        
        返回:
            参数组列表，包含需要L2正则化的参数
        """
        return [
            {
                "params": [p for n, p in self.named_parameters() if "weight" in n],
                "weight_decay": self.l2_reg,
            },
            {
                "params": [p for n, p in self.named_parameters() if "bias" in n],
                "weight_decay": 0.0,
            },
        ]


def create_temporal_encoder(
    encoder_type: str = "baseline",
    num_channels: int = 1,
    sequence_length: int = 1024,
    embedding_dim: int = 96,
    dropout: float = 0.1,
    l2_reg: float = 1e-4,
    return_timesteps: bool = False,
    **kwargs,
) -> nn.Module:
    """
    创建时序编码器的工厂函数。
    
    参数:
        encoder_type: 编码器类型
            - "baseline": 标准3层1D CNN编码器（TemporalEncoder）
            - "multiscale_wavelet": 多路径多尺度小波融合编码器（MultiScaleWaveletTemporalEncoder）
        num_channels: 输入通道数
        sequence_length: 序列长度
        embedding_dim: 输出嵌入维度
        dropout: Dropout比率
        l2_reg: L2正则化系数
        return_timesteps: 是否返回每个时间步的表示（用于TS2Vec等需要时间步表示的场景）
        **kwargs: 其他参数（传递给MultiScaleWaveletTemporalEncoder）
            - num_scales: 多尺度分支数量（默认4）
            - scale_kernels: 各分支的卷积核大小列表（默认[8, 16, 32, 64]）
            - branch_channels: 每个分支的输出通道数（默认32）
            - se_reduction: SE Block的降维比例（默认4）
    
    返回:
        时序编码器实例
    """
    if encoder_type == "baseline":
        return TemporalEncoder(
            num_channels=num_channels,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            dropout=dropout,
            l2_reg=l2_reg,
            return_timesteps=return_timesteps,
        )
    elif encoder_type == "multiscale_wavelet":
        return MultiScaleWaveletTemporalEncoder(
            num_channels=num_channels,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            dropout=dropout,
            l2_reg=l2_reg,
            num_scales=kwargs.get("num_scales", 4),
            scale_kernels=kwargs.get("scale_kernels", None),
            branch_channels=kwargs.get("branch_channels", 32),
            se_reduction=kwargs.get("se_reduction", 4),
            return_timesteps=return_timesteps,
        )
    else:
        raise ValueError(
            f"未知的编码器类型: {encoder_type}。"
            f"可选值: 'baseline', 'multiscale_wavelet'"
        )

