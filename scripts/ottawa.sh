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