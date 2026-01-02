#!/bin/bash

# Ottawa轴承故障诊断数据集训练脚本
# 使用ts2vec进行故障分类

python train.py ottawa ottawa_run \
    --loader ottawa \
    --batch-size 32 \
    --lr 0.001 \
    --repr-dims 320 \
    --max-train-length 3000 \
    --max-threads 8 \
    --seed 42 \
    --epochs 100 \
    --gpu 0 \
    --eval
    --save-every 10

# python train.py ottawa ottawa_run --loader ottawa --batch-size 32 --lr 0.001 --repr-dims 320 --max-train-length 3000 --max-threads 8 --seed 42 --epochs 100 --gpu 0 --eval --save-every 10

# 使用 linear 方式评估
# 注意：如果训练时只使用了部分特征（如仅加速度），需要添加 --feature-columns 参数
# 例如：--feature-columns accelerometer 或 --feature-columns 0
python eval.py training/ottawa__ottawa_run_20251214_130054/model.pkl --eval-protocol linear --dataset ottawa --loader ottawa --gpu 1 --feature-columns accelerometer

# 使用 svm 方式评估
python eval.py training/ottawa__ottawa_run_20251214_130054/model.pkl --eval-protocol svm --dataset ottawa --loader ottawa --gpu 1 --feature-columns accelerometer

# 使用 knn 方式评估
python eval.py training/ottawa__ottawa_run_20251214_130054/model.pkl --eval-protocol knn --dataset ottawa --loader ottawa --gpu 1 --feature-columns accelerometer
