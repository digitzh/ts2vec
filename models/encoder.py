import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
import sys
import os
# 添加项目根目录到路径，以便导入 doc.temporal_encoder
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from doc.temporal_encoder import create_temporal_encoder

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class TSEncoder(nn.Module):
    def __init__(
        self, 
        input_dims, 
        output_dims, 
        hidden_dims=64, 
        depth=10, 
        mask_mode='binomial',
        encoder_type='dilated_conv',
        **encoder_kwargs
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.encoder_type = encoder_type
        
        if encoder_type == 'dilated_conv':
            # 原始TS2Vec编码器
            self.input_fc = nn.Linear(input_dims, hidden_dims)
            self.feature_extractor = DilatedConvEncoder(
                hidden_dims,
                [hidden_dims] * depth + [output_dims],
                kernel_size=3
            )
            self.repr_dropout = nn.Dropout(p=0.1)
        elif encoder_type in ['baseline', 'multiscale_wavelet']:
            # 使用temporal encoder
            # temporal encoder期望输入是 (B, C, L)，输出是 (B, D, L) 或 (B, D)
            # 设置为返回时间步表示，以便TS2Vec进行对比学习
            self.temporal_encoder = create_temporal_encoder(
                encoder_type=encoder_type,
                num_channels=input_dims,
                sequence_length=None,  # 动态长度，不需要固定
                embedding_dim=output_dims,
                dropout=encoder_kwargs.get('dropout', 0.1),
                l2_reg=encoder_kwargs.get('l2_reg', 1e-4),
                return_timesteps=True,  # TS2Vec需要每个时间步的表示
                **{k: v for k, v in encoder_kwargs.items() if k not in ['dropout', 'l2_reg']}
            )
            # temporal encoder不使用input_fc和repr_dropout
            self.input_fc = None
            self.repr_dropout = None
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. "
                f"Supported types: 'dilated_conv', 'baseline', 'multiscale_wavelet'"
            )
        
    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        
        if self.encoder_type == 'dilated_conv':
            # 原始TS2Vec流程
            x = self.input_fc(x)  # B x T x Ch
            
            # generate & apply mask
            if mask is None:
                if self.training:
                    mask = self.mask_mode
                else:
                    mask = 'all_true'
            
            if mask == 'binomial':
                mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
            elif mask == 'continuous':
                mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
            elif mask == 'all_true':
                mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            elif mask == 'all_false':
                mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
            elif mask == 'mask_last':
                mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
                mask[:, -1] = False
            
            mask &= nan_mask
            x[~mask] = 0
            
            # conv encoder
            x = x.transpose(1, 2)  # B x Ch x T
            x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
            x = x.transpose(1, 2)  # B x T x Co
            
        else:
            # temporal encoder流程
            # 输入: B x T x F -> B x F x T
            x = x.transpose(1, 2)  # B x F x T
            
            # 应用mask（如果支持）
            if mask is not None:
                if mask == 'binomial':
                    mask = generate_binomial_mask(x.size(0), x.size(2)).to(x.device)
                elif mask == 'continuous':
                    mask = generate_continuous_mask(x.size(0), x.size(2)).to(x.device)
                elif mask == 'all_true':
                    mask = x.new_full((x.size(0), x.size(2)), True, dtype=torch.bool)
                elif mask == 'all_false':
                    mask = x.new_full((x.size(0), x.size(2)), False, dtype=torch.bool)
                elif mask == 'mask_last':
                    mask = x.new_full((x.size(0), x.size(2)), True, dtype=torch.bool)
                    mask[:, -1] = False
                
                # 扩展mask维度: B x T -> B x 1 x T -> B x F x T
                mask = mask.unsqueeze(1)  # B x 1 x T
                mask = mask & nan_mask.unsqueeze(1)  # 结合nan_mask: B x 1 x T
                mask = mask.expand_as(x)  # B x F x T，扩展到所有特征维度
                x[~mask] = 0
            else:
                # 如果没有指定mask，但训练时可能需要mask
                if self.training and self.mask_mode != 'all_true':
                    if self.mask_mode == 'binomial':
                        mask = generate_binomial_mask(x.size(0), x.size(2)).to(x.device)
                    elif self.mask_mode == 'continuous':
                        mask = generate_continuous_mask(x.size(0), x.size(2)).to(x.device)
                    else:
                        mask = x.new_full((x.size(0), x.size(2)), True, dtype=torch.bool)
                    
                    mask = mask.unsqueeze(1)  # B x 1 x T
                    mask = mask & nan_mask.unsqueeze(1)  # 结合nan_mask: B x 1 x T
                    mask = mask.expand_as(x)  # B x F x T，扩展到所有特征维度
                    x[~mask] = 0
            
            # temporal encoder: B x F x T -> B x D x T
            x = self.temporal_encoder(x)  # B x D x T
            
            # 转换为TS2Vec期望的格式: B x D x T -> B x T x D
            x = x.transpose(1, 2)  # B x T x D
        
        return x
        