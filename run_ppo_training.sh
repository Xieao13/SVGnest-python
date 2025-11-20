#!/bin/bash
# PPO 策略学习训练脚本

# --- 配置 ---
# 使用时间戳创建唯一的输出目录，避免覆盖
timestamp=$(date +%Y%m%d_%H%M%S)
output_dir="./models/ppo_${timestamp}"
logs_dir="./logs/ppo_${timestamp}"

# 为 W&B 运行命名
wandb_run_name="ppo_run_${timestamp}"

# 创建目录
mkdir -p "$output_dir" "$logs_dir"

echo "模型将保存到: $output_dir"
echo "日志将保存到: $logs_dir"

# 如果需要指定GPU，请取消下面的注释
# export CUDA_VISIBLE_DEVICES=0

# 执行 PPO 训练脚本
python3 src/ppo_trainer.py \
    --train_file "./data/placement-0529-ga-20epoch-norotation/train.jsonl" \
    --test_file "./data/placement-0529-ga-20epoch-norotation/test.jsonl" \
    --output_dir "$output_dir" \
    --log_dir "$logs_dir" \
    \
    `# 模型超参数` \
    --input_dim 2 \
    --hidden_dim 128 \
    --max_seq_len 60 \
    \
    `# PPO 核心超参数` \
    --total_timesteps 2000000 \
    --learning_rate 3e-4 \
    --n_steps 2048 \
    --batch_size 64 \
    --n_epochs 10 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_range 0.2 \
    --ent_coef 0.0 \
    --vf_coef 0.5 \
    --max_grad_norm 0.5 \
    \
    `# 环境与奖励配置` \
    --reward_weights 0.65 0.05 0.3 \
    \
    `# 训练与日志配置` \
    --eval_interval 20480 \
    --max_test_instances 1000 \
    --use_wandb \
    --wandb_project "ppo_svgnest" \
    --wandb_run_name "$wandb_run_name"


echo "训练脚本执行完毕。"