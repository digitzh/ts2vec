import torch
import numpy as np
import argparse
import os
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program, pkl_save

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to the saved model file (e.g., training/ottawa__ottawa_run_20251213_214944/model.pkl)')
    parser.add_argument('--dataset', type=str, default='ottawa', help='Dataset name')
    parser.add_argument('--loader', type=str, default='ottawa', help='Data loader')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size (for model config)')
    parser.add_argument('--repr-dims', type=int, default=320, help='Representation dimensions')
    parser.add_argument('--max-train-length', type=int, default=3000, help='Max train length')
    parser.add_argument('--eval-protocol', type=str, default='svm', choices=['linear', 'svm', 'knn', 'mlp'], help='Evaluation protocol')
    parser.add_argument('--feature-columns', type=str, default=None, 
                        help='Feature columns to use (comma-separated). Examples: "accelerometer" or "accelerometer,acoustic" or "0" or "0,1". Must match training configuration.')
    parser.add_argument('--encoder-type', type=str, default='dilated_conv', choices=['dilated_conv', 'baseline', 'multiscale_wavelet'],
                        help='The type of encoder used during training (must match training configuration, defaults to dilated_conv)')
    parser.add_argument('--encoder-dropout', type=float, default=0.1, help='Dropout rate for temporal encoder (must match training)')
    parser.add_argument('--encoder-l2-reg', type=float, default=1e-4, help='L2 regularization coefficient for temporal encoder (must match training)')
    parser.add_argument('--num-scales', type=int, default=4, help='Number of scales for multiscale_wavelet encoder (must match training)')
    parser.add_argument('--branch-channels', type=int, default=32, help='Number of channels per branch for multiscale_wavelet encoder (must match training)')
    parser.add_argument('--se-reduction', type=int, default=4, help='SE Block reduction ratio for multiscale_wavelet encoder (must match training)')
    parser.add_argument('--max-seq-length', type=int, default=None, help='Maximum sequence length to use (truncate sequences if longer). If None, use full sequence.')
    parser.add_argument('--clf-lr', type=float, default=1e-3, help='Initial learning rate for classifier training (default: 1e-3)')
    parser.add_argument('--clf-scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'], 
                        help='Learning rate scheduler for classifier training (default: cosine)')
    parser.add_argument('--clf-epochs', type=int, default=100, help='Number of epochs for classifier training (default: 100)')
    args = parser.parse_args()
    
    # 初始化设备
    device = init_dl_program(args.gpu)
    if isinstance(device, list):
        device = device[0]
    
    # 解析 feature_columns 参数
    feature_columns = None
    if args.feature_columns is not None:
        # 支持逗号分隔的字符串，如 "accelerometer" 或 "accelerometer,acoustic" 或 "0" 或 "0,1"
        parts = [p.strip() for p in args.feature_columns.split(',')]
        if len(parts) == 1:
            # 单个值，可能是字符串或整数
            part = parts[0]
            if part.isdigit():
                # 是数字，转换为整数列表
                feature_columns = [int(part)]
            else:
                # 是字符串，直接使用
                feature_columns = part
        else:
            # 多个值，检查第一个是否是数字
            if parts[0].isdigit():
                # 都是数字，转换为整数列表
                feature_columns = [int(p.strip()) for p in parts]
            else:
                # 都是字符串，使用字符串列表
                feature_columns = parts
    
    print(f'Loading data... ', end='')
    if args.loader == 'ottawa':
        train_data, train_labels, test_data, test_labels = datautils.load_ottawa(
            args.dataset, 
            feature_columns=feature_columns
        )
        task_type = 'classification'
    else:
        raise ValueError(f"Unsupported loader: {args.loader}")
    print('done')
    
    # 截取序列长度（如果指定）
    if args.max_seq_length is not None and train_data.shape[1] > args.max_seq_length:
        print(f'\n截取序列长度: {train_data.shape[1]} -> {args.max_seq_length}')
        train_data = train_data[:, :args.max_seq_length, :]
        test_data = test_data[:, :args.max_seq_length, :]
    
    print(f'\n数据信息:')
    print(f'  训练样本数: {len(train_data)}')
    print(f'  测试样本数: {len(test_data)}')
    print(f'  数据形状: {train_data.shape}')
    if args.feature_columns is not None:
        print(f'  使用的特征列: {args.feature_columns}')
    else:
        print(f'  使用的特征列: 全部5个特征')
    
    # 准备编码器参数（必须与训练时一致）
    encoder_kwargs = {}
    if args.encoder_type in ['baseline', 'multiscale_wavelet']:
        encoder_kwargs['dropout'] = args.encoder_dropout
        encoder_kwargs['l2_reg'] = args.encoder_l2_reg
        if args.encoder_type == 'multiscale_wavelet':
            encoder_kwargs['num_scales'] = args.num_scales
            encoder_kwargs['branch_channels'] = args.branch_channels
            encoder_kwargs['se_reduction'] = args.se_reduction
    
    # 创建模型（使用与训练时相同的配置）
    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        batch_size=args.batch_size,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
        encoder_type=args.encoder_type,
        **encoder_kwargs
    )
    
    # 加载模型
    print(f'\nLoading model from: {args.model_path}')
    model.load(args.model_path)
    print('Model loaded successfully!')
    
    # 运行评估
    print('\n开始评估...')
    print(f'分类器训练参数:')
    print(f'  学习率: {args.clf_lr}')
    print(f'  学习率调度器: {args.clf_scheduler}')
    print(f'  训练轮数: {args.clf_epochs}')
    if task_type == 'classification':
        out, eval_res = tasks.eval_classification(
            model, 
            train_data, 
            train_labels, 
            test_data, 
            test_labels, 
            eval_protocol=args.eval_protocol,
            clf_lr=args.clf_lr,
            clf_scheduler=args.clf_scheduler,
            clf_epochs=args.clf_epochs
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # 保存结果
    model_dir = os.path.dirname(args.model_path)
    pkl_save(f'{model_dir}/out.pkl', out)
    pkl_save(f'{model_dir}/eval_res.pkl', eval_res)
    
    print('\n评估结果:')
    print(eval_res)
    print(f'\n结果已保存到: {model_dir}/')
    print("评估完成！")
