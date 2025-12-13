import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--max-temporal-length', type=int, default=None, help='Maximum temporal length for temporal contrastive loss computation. If None, will be set to min(1000, max_train_length//2) to reduce memory usage (defaults to None)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    # 打印GPU/CPU信息
    print('\n设备信息:')
    if isinstance(device, list):
        device = device[0]  # 如果返回列表，取第一个设备
    
    if device.type == 'cuda':
        device_idx = device.index if device.index is not None else 0
        print(f'  ✓ 使用GPU训练')
        print(f'  GPU设备: {device}')
        try:
            gpu_name = torch.cuda.get_device_name(device_idx)
            print(f'  GPU名称: {gpu_name}')
        except:
            print(f'  GPU名称: 无法获取')
        
        try:
            cuda_version = torch.version.cuda
            if cuda_version:
                print(f'  CUDA版本: {cuda_version}')
        except:
            pass
        
        try:
            cudnn_version = torch.backends.cudnn.version()
            if cudnn_version:
                print(f'  cuDNN版本: {cudnn_version}')
        except:
            pass
        
        # 显存信息
        try:
            props = torch.cuda.get_device_properties(device_idx)
            total_memory = props.total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(device_idx) / 1024**3
            cached_memory = torch.cuda.memory_reserved(device_idx) / 1024**3
            free_memory = total_memory - cached_memory
            print(f'  总显存: {total_memory:.2f} GB')
            print(f'  已分配显存: {allocated_memory:.2f} GB')
            print(f'  已缓存显存: {cached_memory:.2f} GB')
            print(f'  可用显存: {free_memory:.2f} GB')
        except Exception as e:
            print(f'  显存信息: 无法获取 ({str(e)})')
    else:
        print(f'  ⚠ 使用CPU训练（未检测到GPU或GPU不可用）')
        print(f'  CPU设备: {device}')
        if torch.cuda.is_available():
            print(f'  注意: 系统检测到CUDA可用，但当前使用CPU。')
            print(f'        请检查--gpu参数设置（当前值: {args.gpu}）')
    
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        
    elif args.loader == 'ottawa':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_ottawa(args.dataset)
        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data = datautils.gen_ano_train_data(all_train_data)
        
    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data, _, _, _ = datautils.load_UCR('FordA')
        
    else:
        raise ValueError(f"Unknown loader {args.loader}.")
        
        
    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')
    
    # 打印数据信息
    print(f'\n训练数据信息:')
    print(f'  数据形状: {train_data.shape}')
    print(f'  数据大小: {train_data.size:,} 个元素')
    if task_type == 'classification':
        print(f'  训练样本数: {len(train_data)}')
        print(f'  测试样本数: {len(test_data)}')
        print(f'  类别数: {len(np.unique(train_labels))}')
    
    # 自动计算 max_temporal_length 以减少内存使用
    if args.max_temporal_length is None:
        # 根据批次大小和最大训练长度自动设置，避免内存溢出
        # 对于8GB显存，建议 max_temporal_length <= 1000
        max_temporal_length = min(1000, args.max_train_length // 2) if args.max_train_length else 1000
    else:
        max_temporal_length = args.max_temporal_length
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
        max_temporal_length=max_temporal_length
    )

    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # 打印训练配置
    print(f'\n训练配置:')
    print(f'  批次大小: {args.batch_size}')
    print(f'  学习率: {args.lr}')
    print(f'  表示维度: {args.repr_dims}')
    print(f'  最大训练长度: {args.max_train_length}')
    print(f'  时间对比损失最大长度: {max_temporal_length} (用于减少内存使用)')
    if args.epochs is not None:
        print(f'  训练轮数: {args.epochs}')
    elif args.iters is not None:
        print(f'  训练迭代数: {args.iters}')
    else:
        # 根据数据大小自动设置
        default_iters = 200 if train_data.size <= 100000 else 600
        print(f'  自动设置迭代数: {default_iters} (根据数据大小)')
    print(f'  设备: {device}')
    print(f'  输出目录: {run_dir}')
    
    t = time.time()
    
    print(f'\n开始训练...')
    print('=' * 60)
    
    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        else:
            assert False
        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)

    print("Finished.")
