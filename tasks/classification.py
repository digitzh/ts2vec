import numpy as np
import torch
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, f1_score

def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear',
                        clf_lr=1e-3, clf_scheduler='cosine', clf_epochs=100):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    # 使用更小的批次大小进行编码，避免内存溢出
    # 对于评估，使用较小的批次大小（8或16）以减少内存使用
    eval_batch_size = min(8, model.batch_size if hasattr(model, 'batch_size') else 8)
    
    # 获取设备信息
    device = None
    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'net') and hasattr(model.net, 'device'):
        device = model.net.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 如果device是torch.device对象，转换为字符串
    if isinstance(device, torch.device):
        device = str(device)
    
    print(f'  正在编码训练数据 ({len(train_data)} 个样本)...')
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None, batch_size=eval_batch_size)
    print(f'  正在编码测试数据 ({len(test_data)} 个样本)...')
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None, batch_size=eval_batch_size)

    if eval_protocol == 'linear':
        fit_clf = lambda features, y: eval_protocols.fit_lr(
            features, y, device=device, lr=clf_lr, scheduler_type=clf_scheduler, max_iter=clf_epochs)
    elif eval_protocol == 'svm':
        fit_clf = lambda features, y: eval_protocols.fit_svm(
            features, y, device=device, lr=clf_lr, scheduler_type=clf_scheduler, max_iter=clf_epochs)
    elif eval_protocol == 'knn':
        fit_clf = lambda features, y: eval_protocols.fit_knn(features, y, device=device)
    elif eval_protocol == 'mlp':
        fit_clf = lambda features, y: eval_protocols.fit_mlp(
            features, y, device=device, lr=clf_lr, scheduler_type=clf_scheduler, max_iter=clf_epochs)
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    print(f'  正在训练 {eval_protocol.upper()} 分类器...')
    print(f'    训练数据形状: {train_repr.shape}, 类别数: {len(np.unique(train_labels))}')
    print(f'    训练数据内存占用: {train_repr.nbytes / 1024 / 1024:.2f} MB')
    import time
    start_time = time.time()
    clf = fit_clf(train_repr, train_labels)
    elapsed_time = time.time() - start_time
    print(f'  分类器训练完成，总耗时: {elapsed_time:.2f} 秒')
    print(f'  正在评估分类器...')

    acc = clf.score(test_repr, test_labels)
    if eval_protocol == 'svm':
        y_score = clf.decision_function(test_repr)
    else:
        # linear, mlp 和 knn 都使用 predict_proba
        y_score = clf.predict_proba(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    
    # 计算F1-score（Macro和Weighted）
    y_pred = clf.predict(test_repr)
    f1_macro = f1_score(test_labels, y_pred, average='macro')
    f1_weighted = f1_score(test_labels, y_pred, average='weighted')
    
    return y_score, { 'acc': acc, 'auprc': auprc, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted }
