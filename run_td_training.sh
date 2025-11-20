#!/bin/bash
# 时序差分策略学习训练脚本
# 创建目录
timestamp=$(date +%Y%m%d%H%M%S)
output_dir='./models/$timestamp'
logs_dir='./logs/$timestamp'
mkdir -p $output_dir $logs_dir
# 运行训练
# export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
# export CUDA_VISIBLE_DEVICES=4
python3 src/td_trainer.py \
    --train_file "./data/placement-0529-ga-20epoch-norotation/train.jsonl" \
    --test_file "./data/placement-0529-ga-20epoch-norotation/test.jsonl" \
    --max_train_instances 50000 \
    --max_test_instances 2000 \
    --num_epochs 100 \
    --reward_weights 0.65 0.05 0.3 \
    --use_baseline False \
    --use_wandb True\
    --wandb_project "td_svgnest" \
    --input_dim 2 \
    --hidden_dim 128 \
    --max_seq_len 60 \
    --learning_rate 3e-4 \
    --clip_grad_norm 1.0 \
    --output_dir $output_dir \
    --log_dir $logs_dir \
    --save_interval_steps 10000