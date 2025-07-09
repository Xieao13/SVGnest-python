#!/bin/bash

# 时序差分策略学习训练脚本

# 创建目录
mkdir -p ./models ./logs

# 运行训练
python src/td_trainer.py \
    --train_file "./data/placement-0529-ga-20epoch-norotation/train.jsonl" \
    --test_file "./data/placement-0529-ga-20epoch-norotation/test.jsonl" \
    --max_train_instances 50000 \
    --max_test_instances 2000 \
    --num_epochs 100 \
    --use_wandb \
    --wandb_project "td_svgnest"