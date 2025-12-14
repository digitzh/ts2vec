import torch
from torch import nn
import torch.nn.functional as F

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0, max_temporal_length=1000):
    """
    计算分层对比损失。
    
    Args:
        z1, z2: 形状为 (B, T, C) 的张量
        alpha: 实例对比损失的权重
        temporal_unit: 开始计算时间对比损失的最小层级
        max_temporal_length: temporal_contrastive_loss 中使用的最大时间步数
    """
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2, max_temporal_length)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2, max_temporal_length=1000):
    """
    计算时间对比损失。
    
    Args:
        z1, z2: 形状为 (B, T, C) 的张量
        max_temporal_length: 最大时间步数，超过此长度将进行采样以减少内存使用。
                            如果设置为 None，则不进行采样限制（适用于显存充足的情况）
    """
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    
    # 如果设置了 max_temporal_length 且序列太长，进行采样以减少内存使用
    if max_temporal_length is not None and T > max_temporal_length:
        # 随机采样 max_temporal_length 个时间步
        indices = torch.randperm(T, device=z1.device)[:max_temporal_length].sort()[0]
        z1 = z1[:, indices]
        z2 = z2[:, indices]
        T = max_temporal_length
    
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss
