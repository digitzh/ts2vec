#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查看 .pkl 文件内容的工具脚本
用法: python view_pkl.py <pkl_file_path>
"""

import argparse
import numpy as np
from utils import pkl_load

def format_value(value, indent=0):
    """格式化显示值"""
    indent_str = "  " * indent
    
    if isinstance(value, np.ndarray):
        return f"{indent_str}ndarray: shape={value.shape}, dtype={value.dtype}\n" \
               f"{indent_str}  min={value.min():.6f}, max={value.max():.6f}, mean={value.mean():.6f}, std={value.std():.6f}"
    elif isinstance(value, dict):
        lines = [f"{indent_str}dict with {len(value)} keys:"]
        for k, v in value.items():
            lines.append(f"{indent_str}  '{k}': {format_value(v, indent+2)}")
        return "\n".join(lines)
    elif isinstance(value, (list, tuple)):
        lines = [f"{indent_str}{type(value).__name__} with {len(value)} items:"]
        if len(value) > 0:
            lines.append(f"{indent_str}  First item: {format_value(value[0], indent+2)}")
            if len(value) > 1:
                lines.append(f"{indent_str}  ... ({len(value)-1} more items)")
        return "\n".join(lines)
    elif isinstance(value, (int, float)):
        return f"{value}"
    elif isinstance(value, str):
        return f"'{value}'"
    else:
        return f"{type(value).__name__}: {str(value)[:100]}"

def view_pkl(file_path, show_details=True, show_sample=True, max_samples=5):
    """查看 pkl 文件内容"""
    try:
        print(f"正在加载: {file_path}")
        data = pkl_load(file_path)
        print("加载成功!\n")
        print("=" * 60)
        print("文件内容:")
        print("=" * 60)
        
        if isinstance(data, dict):
            print(f"类型: 字典 (包含 {len(data)} 个键)")
            print("\n键值对:")
            for key, value in data.items():
                print(f"\n  [{key}]")
                print(format_value(value, indent=2))
                
                # 如果是数组，显示样本
                if isinstance(value, np.ndarray) and show_sample:
                    if value.ndim <= 2 and value.size <= 100:
                        print(f"\n    完整数据:\n{value}")
                    elif value.ndim == 1:
                        print(f"\n    前 {min(max_samples, len(value))} 个值: {value[:max_samples]}")
                    elif value.ndim == 2:
                        print(f"\n    前 {min(max_samples, value.shape[0])} 行:\n{value[:max_samples]}")
        elif isinstance(data, np.ndarray):
            print(f"类型: NumPy 数组")
            print(format_value(data, indent=0))
            if show_sample:
                if data.ndim <= 2 and data.size <= 100:
                    print(f"\n完整数据:\n{data}")
                elif data.ndim == 1:
                    print(f"\n前 {min(max_samples, len(data))} 个值: {data[:max_samples]}")
                elif data.ndim == 2:
                    print(f"\n前 {min(max_samples, data.shape[0])} 行:\n{data[:max_samples]}")
        else:
            print(f"类型: {type(data).__name__}")
            print(format_value(data, indent=0))
            if show_sample and isinstance(data, (list, tuple)) and len(data) > 0:
                print(f"\n前 {min(max_samples, len(data))} 个元素:")
                for i, item in enumerate(data[:max_samples]):
                    print(f"  [{i}]: {item}")
        
        print("\n" + "=" * 60)
        
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
    except Exception as e:
        print(f"错误: 无法加载文件 '{file_path}'")
        print(f"详细信息: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='查看 .pkl 文件内容')
    parser.add_argument('file_path', help='pkl 文件路径')
    parser.add_argument('--no-details', action='store_true', help='不显示详细信息')
    parser.add_argument('--no-sample', action='store_true', help='不显示样本数据')
    parser.add_argument('--max-samples', type=int, default=5, help='显示的最大样本数 (默认: 5)')
    
    args = parser.parse_args()
    
    view_pkl(
        args.file_path,
        show_details=not args.no_details,
        show_sample=not args.no_sample,
        max_samples=args.max_samples
    )


