import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    
    
def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, univar=False):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
        
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data


def load_ottawa(dataset='ottawa', train_ratio=0.8, segment_length=None, segment_stride=None, random_seed=42, feature_columns=None):
    """
    加载Ottawa轴承故障数据集
    
    参数:
        dataset: 数据集名称（默认'ottawa'，实际不使用）
        train_ratio: 训练集比例（默认0.8）
        segment_length: 如果指定，将每个长序列分段为指定长度的子序列
        segment_stride: 分段时的步长，如果为None则等于segment_length（无重叠）
        random_seed: 随机种子，用于确保可重复性（默认42）
        feature_columns: 要使用的特征列，可以是：
            - None: 使用所有5个特征（默认）
            - 字符串: 'accelerometer', 'acoustic', 'speed', 'load', 'temperature' 之一
            - 整数列表: 列索引列表，如 [0] 表示只使用加速度，[0,1] 表示使用加速度和声学
            - 字符串列表: 列名列表，如 ['Accelerometer', 'Acoustic']
    
    返回:
        train_data: (n_train_instances, n_timestamps, n_features) 训练数据
        train_labels: (n_train_instances,) 训练标签
        test_data: (n_test_instances, n_timestamps, n_features) 测试数据
        test_labels: (n_test_instances,) 测试标签
    """
    # 设置随机种子以确保可重复性
    np.random.seed(random_seed)
    
    base_dir = 'datasets/csv'
    
    # 定义特征列映射
    feature_mapping = {
        'accelerometer': 0,
        'acoustic': 1,
        'speed': 2,
        'load': 3,
        'temperature': 4,
        'Accelerometer': 0,
        'Acoustic': 1,
        'Speed': 2,
        'Load': 3,
        'Temperature Difference': 4,
        'Temperature': 4
    }
    
    # 解析 feature_columns 参数
    column_indices = None
    if feature_columns is not None:
        if isinstance(feature_columns, str):
            # 单个字符串，转换为列表
            feature_columns = [feature_columns]
        
        if isinstance(feature_columns, list):
            if len(feature_columns) == 0:
                raise ValueError("feature_columns 不能为空列表")
            
            # 检查第一个元素是字符串还是整数
            if isinstance(feature_columns[0], str):
                # 字符串列表，转换为列索引
                column_indices = []
                for col in feature_columns:
                    col_lower = col.lower()
                    if col_lower in feature_mapping:
                        column_indices.append(feature_mapping[col_lower])
                    elif col in feature_mapping:
                        column_indices.append(feature_mapping[col])
                    else:
                        raise ValueError(f"未知的特征列名: {col}。可用选项: {list(set([k for k in feature_mapping.keys() if k.lower() in ['accelerometer', 'acoustic', 'speed', 'load', 'temperature']]))}")
            elif isinstance(feature_columns[0], int):
                # 整数列表，直接使用
                column_indices = feature_columns
                if not all(0 <= idx < 5 for idx in column_indices):
                    raise ValueError(f"列索引必须在 [0, 4] 范围内，当前值: {column_indices}")
            else:
                raise ValueError(f"feature_columns 的元素必须是字符串或整数，当前类型: {type(feature_columns[0])}")
        else:
            raise ValueError(f"feature_columns 必须是字符串、整数列表或字符串列表，当前类型: {type(feature_columns)}")
    
    # 定义类别映射：H=0, I=1, O=2, B=3, C=4
    class_mapping = {
        'H': 0,  # Healthy
        'I': 1,  # Inner Race Faults
        'O': 2,  # Outer Race Faults
        'B': 3,  # Ball Faults
        'C': 4   # Cage Faults
    }
    
    # 定义类别文件夹
    class_folders = {
        'H': '1_Healthy',
        'I': '2_Inner_Race_Faults',
        'O': '3_Outer_Race_Faults',
        'B': '4_Ball_Faults',
        'C': '5_Cage_Faults'
    }
    
    all_data = []
    all_labels = []
    
    # 遍历所有类别
    for class_code, class_folder in class_folders.items():
        folder_path = os.path.join(base_dir, class_folder)
        if not os.path.exists(folder_path):
            continue
            
        # 获取该类别下的所有CSV文件
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        csv_files.sort()  # 排序以确保一致性
        
        # 为每个文件加载数据
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            try:
                # 读取CSV文件（跳过标题行）
                df = pd.read_csv(file_path, header=0)
                
                # 提取数据（5列：加速度、声学、转速、负载、温度差）
                data = df.values.astype(np.float64)
                
                # 如果指定了特征列，只选择这些列
                if column_indices is not None:
                    data = data[:, column_indices]
                
                # 如果指定了分段长度，将长序列分段
                if segment_length is not None:
                    stride = segment_stride if segment_stride is not None else segment_length
                    for i in range(0, len(data) - segment_length + 1, stride):
                        segment = data[i:i+segment_length]
                        all_data.append(segment)
                        all_labels.append(class_mapping[class_code])
                else:
                    # 使用完整序列
                    all_data.append(data)
                    all_labels.append(class_mapping[class_code])
                    
            except Exception as e:
                print(f"警告: 无法加载文件 {file_path}: {e}")
                continue
    
    if len(all_data) == 0:
        raise ValueError("未找到任何数据文件，请检查数据集路径")
    
    # 转换为numpy数组
    # 如果所有序列长度相同，直接stack；否则需要padding
    lengths = [len(d) for d in all_data]
    max_length = max(lengths)
    min_length = min(lengths)
    
    if max_length == min_length:
        # 所有序列长度相同
        all_data = np.array(all_data)  # (n_instances, n_timestamps, n_features)
    else:
        # 需要padding到最大长度
        print(f"警告: 序列长度不一致 (最小: {min_length}, 最大: {max_length})，将padding到最大长度")
        padded_data = []
        for d in all_data:
            if len(d) < max_length:
                # 使用nan padding
                padding = np.full((max_length - len(d), d.shape[1]), np.nan)
                d = np.vstack([d, padding])
            padded_data.append(d)
        all_data = np.array(padded_data)
    
    all_labels = np.array(all_labels)
    
    # 数据标准化
    # 在整个数据集上计算均值和标准差（忽略nan值）
    data_flat = all_data.reshape(-1, all_data.shape[-1])
    scaler = StandardScaler()
    scaler.fit(data_flat[~np.isnan(data_flat).any(axis=1)])
    
    # 标准化每个样本
    for i in range(len(all_data)):
        valid_mask = ~np.isnan(all_data[i]).any(axis=1)
        if valid_mask.sum() > 0:
            all_data[i][valid_mask] = scaler.transform(all_data[i][valid_mask])
    
    # 划分训练集和测试集
    # 按类别分层划分
    train_data_list = []
    train_labels_list = []
    test_data_list = []
    test_labels_list = []
    
    for class_id in range(len(class_mapping)):
        class_mask = all_labels == class_id
        class_data = all_data[class_mask]
        class_labels = all_labels[class_mask]
        
        n_samples = len(class_data)
        n_train = int(n_samples * train_ratio)
        
        # 随机打乱
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        train_data_list.append(class_data[train_indices])
        train_labels_list.append(class_labels[train_indices])
        test_data_list.append(class_data[test_indices])
        test_labels_list.append(class_labels[test_indices])
    
    # 合并所有类别的数据
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    test_data = np.concatenate(test_data_list, axis=0)
    test_labels = np.concatenate(test_labels_list, axis=0)
    
    # 再次打乱训练集和测试集
    train_indices = np.arange(len(train_data))
    test_indices = np.arange(len(test_data))
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    train_data = train_data[train_indices]
    train_labels = train_labels[train_indices]
    test_data = test_data[test_indices]
    test_labels = test_labels[test_indices]
    
    # 生成特征名称用于显示
    feature_names = ['加速度', '声学', '转速', '负载', '温度差']
    if column_indices is not None:
        selected_features = [feature_names[i] for i in column_indices]
        feature_info = f"{len(column_indices)}个特征: {', '.join(selected_features)}"
    else:
        feature_info = f"5个特征: {', '.join(feature_names)}"
    
    print(f"数据集加载完成:")
    print(f"  训练集: {len(train_data)} 个样本")
    print(f"  测试集: {len(test_data)} 个样本")
    print(f"  序列长度: {train_data.shape[1]}")
    print(f"  特征数: {train_data.shape[2]} ({feature_info})")
    print(f"  类别数: {len(class_mapping)}")
    print(f"  类别分布 - 训练集: {np.bincount(train_labels)}, 测试集: {np.bincount(test_labels)}")
    
    return train_data, train_labels, test_data, test_labels