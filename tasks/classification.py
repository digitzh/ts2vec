import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, f1_score

def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    # 使用更小的批次大小进行编码，避免内存溢出
    # 对于评估，使用较小的批次大小（8或16）以减少内存使用
    eval_batch_size = min(8, model.batch_size if hasattr(model, 'batch_size') else 8)
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None, batch_size=eval_batch_size)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None, batch_size=eval_batch_size)

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    clf = fit_clf(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)
    if eval_protocol == 'svm':
        y_score = clf.decision_function(test_repr)
    else:
        # linear (LogisticRegression) 和 knn (KNeighborsClassifier) 都使用 predict_proba
        y_score = clf.predict_proba(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    
    # 计算F1-score（Macro和Weighted）
    y_pred = clf.predict(test_repr)
    f1_macro = f1_score(test_labels, y_pred, average='macro')
    f1_weighted = f1_score(test_labels, y_pred, average='weighted')
    
    return y_score, { 'acc': acc, 'auprc': auprc, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted }
