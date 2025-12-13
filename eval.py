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
    parser.add_argument('--eval-protocol', type=str, default='svm', choices=['linear', 'svm', 'knn'], help='Evaluation protocol')
    args = parser.parse_args()
    
    # 初始化设备
    device = init_dl_program(args.gpu)
    if isinstance(device, list):
        device = device[0]
    
    print(f'Loading data... ', end='')
    if args.loader == 'ottawa':
        train_data, train_labels, test_data, test_labels = datautils.load_ottawa(args.dataset)
        task_type = 'classification'
    else:
        raise ValueError(f"Unsupported loader: {args.loader}")
    print('done')
    
    print(f'\n数据信息:')
    print(f'  训练样本数: {len(train_data)}')
    print(f'  测试样本数: {len(test_data)}')
    print(f'  数据形状: {train_data.shape}')
    
    # 创建模型（使用与训练时相同的配置）
    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        batch_size=args.batch_size,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    
    # 加载模型
    print(f'\nLoading model from: {args.model_path}')
    model.load(args.model_path)
    print('Model loaded successfully!')
    
    # 运行评估
    print('\n开始评估...')
    if task_type == 'classification':
        out, eval_res = tasks.eval_classification(
            model, 
            train_data, 
            train_labels, 
            test_data, 
            test_labels, 
            eval_protocol=args.eval_protocol
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