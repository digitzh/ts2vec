#!/usr/bin/env python
"""
测试Ottawa数据加载器
用于验证数据加载是否正常工作
"""

import sys
import numpy as np
import datautils

def test_load_ottawa():
    """测试load_ottawa函数"""
    print("=" * 60)
    print("测试Ottawa数据加载器")
    print("=" * 60)
    
    try:
        # 测试基本加载
        print("\n1. 测试基本数据加载...")
        train_data, train_labels, test_data, test_labels = datautils.load_ottawa()
        
        # 检查数据形状
        print(f"\n数据形状检查:")
        print(f"  训练数据形状: {train_data.shape}")
        print(f"  训练标签形状: {train_labels.shape}")
        print(f"  测试数据形状: {test_data.shape}")
        print(f"  测试标签形状: {test_labels.shape}")
        
        # 检查数据类型
        assert train_data.dtype == np.float64, "训练数据类型应为float64"
        assert test_data.dtype == np.float64, "测试数据类型应为float64"
        assert train_labels.dtype in [np.int32, np.int64], "训练标签应为整数类型"
        assert test_labels.dtype in [np.int32, np.int64], "测试标签应为整数类型"
        print("  ✓ 数据类型检查通过")
        
        # 检查数据维度
        assert len(train_data.shape) == 3, "训练数据应为3维 (n_samples, n_timestamps, n_features)"
        assert len(test_data.shape) == 3, "测试数据应为3维 (n_samples, n_timestamps, n_features)"
        assert train_data.shape[2] == 5, "特征数应为5 (加速度、声学、转速、负载、温度差)"
        assert test_data.shape[2] == 5, "特征数应为5"
        print("  ✓ 数据维度检查通过")
        
        # 检查标签范围
        unique_labels = np.unique(np.concatenate([train_labels, test_labels]))
        assert len(unique_labels) == 5, "应该有5个类别"
        assert np.all(unique_labels == np.arange(5)), "标签应为0-4"
        print(f"  ✓ 标签检查通过 (类别: {unique_labels})")
        
        # 检查数据划分
        assert len(train_data) > 0, "训练集不应为空"
        assert len(test_data) > 0, "测试集不应为空"
        train_ratio = len(train_data) / (len(train_data) + len(test_data))
        assert 0.6 < train_ratio < 0.8, f"训练集比例应在0.6-0.8之间，实际为{train_ratio:.2f}"
        print(f"  ✓ 数据划分检查通过 (训练集比例: {train_ratio:.2f})")
        
        # 检查数据值
        assert not np.all(np.isnan(train_data)), "训练数据不应全为NaN"
        assert not np.all(np.isnan(test_data)), "测试数据不应全为NaN"
        print("  ✓ 数据值检查通过")
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！数据加载器工作正常。")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_load_ottawa()
    sys.exit(0 if success else 1)

