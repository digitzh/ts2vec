"""
PyTorch 版本的分类器实现
替代 sklearn 的 Linear, SVM, KNN 分类器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


class LinearClassifier(nn.Module):
    """PyTorch 版本的逻辑回归分类器"""
    
    def __init__(self, input_dim, num_classes, device='cuda'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes).to(device)
        self.scaler = StandardScaler()
        
    def fit(self, X, y, max_iter=1000, lr=0.01, batch_size=256, scheduler_type='cosine'):
        """训练分类器
        
        Args:
            X: 训练特征
            y: 训练标签
            max_iter: 最大迭代次数（epochs）
            lr: 初始学习率
            batch_size: 批次大小
            scheduler_type: 学习率调度器类型 ('cosine', 'step', 'none')
        """
        X = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # 设置学习率调度器
        scheduler = None
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max_iter//3, gamma=0.1)
        # 'none' 时不使用调度器
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.train()
        for epoch in range(max_iter):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = self.linear(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
            
            if epoch % 100 == 0 and epoch > 0:
                avg_loss = total_loss / len(loader)
                current_lr = optimizer.param_groups[0]['lr']
                print(f'    Epoch {epoch}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
        
        return self
    
    def predict(self, X):
        """预测类别"""
        self.eval()
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.linear(X_tensor)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X):
        """预测概率"""
        self.eval()
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.linear(X_tensor)
            probs = F.softmax(logits, dim=1)
        
        return probs.cpu().numpy()
    
    def decision_function(self, X):
        """决策函数（用于兼容性）"""
        self.eval()
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.linear(X_tensor)
        
        return logits.cpu().numpy()
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class SVMClassifier(nn.Module):
    """PyTorch 版本的 SVM 分类器（使用 Hinge Loss）"""
    
    def __init__(self, input_dim, num_classes, C=1.0, device='cuda'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.C = C
        self.linear = nn.Linear(input_dim, num_classes).to(device)
        self.scaler = StandardScaler()
        
    def fit(self, X, y, max_iter=1000, lr=0.001, batch_size=256, scheduler_type='cosine'):
        """训练 SVM 分类器
        
        Args:
            X: 训练特征
            y: 训练标签
            max_iter: 最大迭代次数（epochs）
            lr: 初始学习率
            batch_size: 批次大小
            scheduler_type: 学习率调度器类型 ('cosine', 'step', 'none')
        """
        X = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr, weight_decay=1.0/self.C)
        
        # 设置学习率调度器
        scheduler = None
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max_iter//3, gamma=0.1)
        # 'none' 时不使用调度器
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.train()
        for epoch in range(max_iter):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = self.linear(batch_X)
                
                # Multi-class Hinge Loss
                one_hot = F.one_hot(batch_y, num_classes=self.num_classes).float()
                correct_logits = (logits * one_hot).sum(dim=1)
                wrong_logits = (logits * (1 - one_hot) - one_hot * 1e9).max(dim=1)[0]
                loss = F.relu(1 + wrong_logits - correct_logits).mean()
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
            
            if epoch % 100 == 0 and epoch > 0:
                avg_loss = total_loss / len(loader)
                current_lr = optimizer.param_groups[0]['lr']
                print(f'    Epoch {epoch}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
        
        return self
    
    def predict(self, X):
        """预测类别"""
        self.eval()
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.linear(X_tensor)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def decision_function(self, X):
        """决策函数"""
        self.eval()
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.linear(X_tensor)
        
        return logits.cpu().numpy()
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class KNNClassifier:
    """PyTorch 版本的 KNN 分类器（使用 GPU 加速）"""
    
    def __init__(self, n_neighbors=1, device='cuda'):
        self.n_neighbors = n_neighbors
        self.device = device
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """训练 KNN（实际上只是存储训练数据）"""
        X = self.scaler.fit_transform(X)
        self.X_train = torch.FloatTensor(X).to(self.device)
        self.y_train = torch.LongTensor(y).to(self.device)
        return self
    
    def predict(self, X):
        """预测类别"""
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # 计算距离矩阵
        distances = torch.cdist(X_tensor, self.X_train)
        
        # 找到 k 个最近邻
        _, indices = torch.topk(distances, k=self.n_neighbors, dim=1, largest=False)
        
        # 获取最近邻的标签
        neighbor_labels = self.y_train[indices]
        
        # 对于 k=1，直接返回；对于 k>1，返回众数
        if self.n_neighbors == 1:
            predictions = neighbor_labels.squeeze(1)
        else:
            # 使用 mode（众数）
            predictions = torch.mode(neighbor_labels, dim=1)[0]
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X):
        """预测概率（基于最近邻的投票）"""
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # 计算距离矩阵
        distances = torch.cdist(X_tensor, self.X_train)
        
        # 找到 k 个最近邻
        _, indices = torch.topk(distances, k=self.n_neighbors, dim=1, largest=False)
        
        # 获取最近邻的标签
        neighbor_labels = self.y_train[indices]
        
        # 计算每个类别的投票数
        num_classes = len(torch.unique(self.y_train))
        batch_size = X_tensor.size(0)
        probs = torch.zeros(batch_size, num_classes, device=self.device)
        
        for i in range(batch_size):
            labels = neighbor_labels[i]
            for label in labels:
                probs[i, label] += 1
        
        # 归一化
        probs = probs / self.n_neighbors
        
        return probs.cpu().numpy()
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def fit_linear(features, y, device='cuda', max_iter=1000, MAX_SAMPLES=100000, lr=1e-3, scheduler_type='cosine'):
    """训练线性分类器
    
    Args:
        features: 训练特征
        y: 训练标签
        device: 设备 ('cuda' 或 'cpu')
        max_iter: 最大迭代次数（epochs）
        MAX_SAMPLES: 最大样本数（超过则采样）
        lr: 初始学习率
        scheduler_type: 学习率调度器类型 ('cosine', 'step', 'none')
    """
    # 如果样本数过多，进行采样
    if features.shape[0] > MAX_SAMPLES:
        from sklearn.model_selection import train_test_split
        features, _, y, _ = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        print(f'    样本数过多，采样到 {MAX_SAMPLES} 个样本')
    
    num_classes = len(np.unique(y))
    input_dim = features.shape[1]
    
    clf = LinearClassifier(input_dim, num_classes, device=device)
    clf.fit(features, y, max_iter=max_iter, lr=lr, scheduler_type=scheduler_type)
    return clf


def fit_svm(features, y, device='cuda', C=None, max_iter=1000, MAX_SAMPLES=10000, lr=1e-3, scheduler_type='cosine'):
    """训练 SVM 分类器，支持C参数网格搜索
    
    Args:
        features: 训练特征
        y: 训练标签
        device: 设备 ('cuda' 或 'cpu')
        C: SVM正则化参数（None时进行网格搜索）
        max_iter: 最大迭代次数（epochs）
        MAX_SAMPLES: 最大样本数（超过则采样）
        lr: 初始学习率
        scheduler_type: 学习率调度器类型 ('cosine', 'step', 'none')
    """
    num_classes = len(np.unique(y))
    input_dim = features.shape[1]
    train_size = features.shape[0]
    
    # 如果样本数过多，进行采样
    if train_size > MAX_SAMPLES:
        from sklearn.model_selection import train_test_split
        features, _, y, _ = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        train_size = len(features)
        print(f'    样本数过多，采样到 {MAX_SAMPLES} 个样本')
    
    # 如果样本数很少，使用简单训练
    if train_size // num_classes < 5 or train_size < 50:
        print(f'    使用简单SVM训练 (样本数: {train_size}, 类别数: {num_classes})')
        if C is None:
            C = np.inf
        clf = SVMClassifier(input_dim, num_classes, C=C, device=device)
        clf.fit(features, y, max_iter=max_iter, lr=lr, scheduler_type=scheduler_type)
        return clf
    else:
        # 使用网格搜索选择最佳C值
        if C is None:
            # 默认C值列表
            C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf]
        else:
            C_values = [C] if not isinstance(C, list) else C
        
        print(f'    开始SVM网格搜索: {len(C_values)} 个C参数')
        print(f'    样本数: {train_size}, 特征维度: {input_dim}, 类别数: {num_classes}')
        
        best_clf = None
        best_score = -np.inf
        best_C = None
        
        # 使用交叉验证（简化版：使用部分数据作为验证集）
        from sklearn.model_selection import train_test_split
        train_features, val_features, train_y, val_y = train_test_split(
            features, y, test_size=0.2, random_state=0, stratify=y
        )
        
        for c_val in C_values:
            clf = SVMClassifier(input_dim, num_classes, C=c_val, device=device)
            clf.fit(train_features, train_y, max_iter=max_iter, lr=lr, scheduler_type=scheduler_type)
            score = clf.score(val_features, val_y)
            if score > best_score:
                best_score = score
                best_clf = clf
                best_C = c_val
        
        print(f'    最佳参数: C={best_C}, 验证准确率: {best_score:.4f}')
        
        # 用全部数据重新训练最佳模型
        best_clf = SVMClassifier(input_dim, num_classes, C=best_C, device=device)
        best_clf.fit(features, y, max_iter=max_iter, lr=lr, scheduler_type=scheduler_type)
        return best_clf


def fit_knn(features, y, device='cuda', n_neighbors=1):
    """训练 KNN 分类器"""
    clf = KNNClassifier(n_neighbors=n_neighbors, device=device)
    clf.fit(features, y)
    return clf


class MLPClassifier(nn.Module):
    """PyTorch 版本的 MLP 分类器（多层感知机）"""
    
    def __init__(self, input_dim, num_classes, classifier_hidden_dim=128, dropout=0.1, device='cuda'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        ).to(device)
        self.scaler = StandardScaler()
        
    def fit(self, X, y, max_iter=1000, lr=0.01, batch_size=256, scheduler_type='cosine'):
        """训练 MLP 分类器
        
        Args:
            X: 训练特征
            y: 训练标签
            max_iter: 最大迭代次数（epochs）
            lr: 初始学习率
            batch_size: 批次大小
            scheduler_type: 学习率调度器类型 ('cosine', 'step', 'none')
        """
        X = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # 设置学习率调度器
        scheduler = None
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max_iter//3, gamma=0.1)
        # 'none' 时不使用调度器
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.train()
        for epoch in range(max_iter):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = self.classifier(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
            
            if epoch % 100 == 0 and epoch > 0:
                avg_loss = total_loss / len(loader)
                current_lr = optimizer.param_groups[0]['lr']
                print(f'    Epoch {epoch}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
        
        return self
    
    def predict(self, X):
        """预测类别"""
        self.eval()
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.classifier(X_tensor)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X):
        """预测概率"""
        self.eval()
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.classifier(X_tensor)
            probs = F.softmax(logits, dim=1)
        
        return probs.cpu().numpy()
    
    def decision_function(self, X):
        """决策函数（用于兼容性）"""
        self.eval()
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.classifier(X_tensor)
        
        return logits.cpu().numpy()
    
    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def fit_mlp(features, y, device='cuda', max_iter=1000, MAX_SAMPLES=100000, lr=1e-3, scheduler_type='cosine', 
             classifier_hidden_dim=128, dropout=0.1):
    """训练 MLP 分类器
    
    Args:
        features: 训练特征
        y: 训练标签
        device: 设备 ('cuda' 或 'cpu')
        max_iter: 最大迭代次数（epochs）
        MAX_SAMPLES: 最大样本数（超过则采样）
        lr: 初始学习率
        scheduler_type: 学习率调度器类型 ('cosine', 'step', 'none')
        classifier_hidden_dim: MLP隐藏层维度
        dropout: Dropout比率
    """
    # 如果样本数过多，进行采样
    if features.shape[0] > MAX_SAMPLES:
        from sklearn.model_selection import train_test_split
        features, _, y, _ = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        print(f'    样本数过多，采样到 {MAX_SAMPLES} 个样本')
    
    num_classes = len(np.unique(y))
    input_dim = features.shape[1]
    
    clf = MLPClassifier(input_dim, num_classes, classifier_hidden_dim=classifier_hidden_dim, 
                        dropout=dropout, device=device)
    clf.fit(features, y, max_iter=max_iter, lr=lr, scheduler_type=scheduler_type)
    return clf

