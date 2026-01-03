import numpy as np
import time
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.classifiers import fit_linear as fit_linear_pytorch, fit_svm as fit_svm_pytorch, fit_knn as fit_knn_pytorch, fit_mlp as fit_mlp_pytorch

def fit_svm(features, y, device=None, MAX_SAMPLES=10000, lr=1e-3, scheduler_type='cosine', max_iter=100):
    """使用PyTorch版本的SVM分类器"""
    # 自动检测设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif isinstance(device, torch.device):
        device = str(device)
    
    nb_classes = len(np.unique(y))
    train_size = features.shape[0]
    feature_dim = features.shape[1]
    
    if feature_dim > 10000:
        print(f'    警告: 特征维度很大 ({feature_dim})，SVM训练可能会很慢')
    
    print(f'    使用PyTorch SVM分类器 (设备: {device})')
    print(f'    样本数: {train_size}, 特征维度: {feature_dim}, 类别数: {nb_classes}')
    
    start_time = time.time()
    clf = fit_svm_pytorch(features, y, device=device, MAX_SAMPLES=MAX_SAMPLES, 
                          lr=lr, scheduler_type=scheduler_type, max_iter=max_iter)
    elapsed_time = time.time() - start_time
    print(f'    SVM训练完成，耗时: {elapsed_time:.2f} 秒')
    
    return clf

def fit_lr(features, y, device=None, MAX_SAMPLES=100000, lr=1e-3, scheduler_type='cosine', max_iter=100):
    """使用PyTorch版本的线性分类器（逻辑回归）"""
    # 自动检测设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif isinstance(device, torch.device):
        device = str(device)
    
    print(f'    使用PyTorch线性分类器 (设备: {device})')
    clf = fit_linear_pytorch(features, y, device=device, MAX_SAMPLES=MAX_SAMPLES,
                             lr=lr, scheduler_type=scheduler_type, max_iter=max_iter)
    return clf

def fit_knn(features, y, device=None):
    """使用PyTorch版本的KNN分类器"""
    # 自动检测设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif isinstance(device, torch.device):
        device = str(device)
    
    print(f'    使用PyTorch KNN分类器 (设备: {device})')
    clf = fit_knn_pytorch(features, y, device=device, n_neighbors=1)
    return clf

def fit_mlp(features, y, device=None, MAX_SAMPLES=100000, lr=1e-3, scheduler_type='cosine', max_iter=100):
    """使用PyTorch版本的MLP分类器"""
    # 自动检测设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif isinstance(device, torch.device):
        device = str(device)
    
    print(f'    使用PyTorch MLP分类器 (设备: {device})')
    clf = fit_mlp_pytorch(features, y, device=device, MAX_SAMPLES=MAX_SAMPLES,
                          lr=lr, scheduler_type=scheduler_type, max_iter=max_iter)
    return clf

def fit_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]
    
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]
    
    lr = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)
    return lr
