#!/bin/bash
# 并行化 PPO 策略学习训练脚本

# --- 配置 ---
timestamp=$(date +%Y%m%d_%H%M%S)
output_dir="./models/ppo_parallel_${timestamp}"
logs_dir="./logs/ppo_parallel_${timestamp}"
wandb_run_name="ppo_parallel_run_${timestamp}"
num_parallel_envs=8 # <--- 在这里设置并行环境数量

mkdir -p "$output_dir" "$logs_dir"

echo "模型将保存到: $output_dir"
echo "日志将保存到: $logs_dir"

# --- 运行训练 ---
python3 src/ppo_trainer_parallel.py \
    --train_file "./data/placement-0529-ga-20epoch-norotation/train.jsonl" \
    --test_file "./data/placement-0529-ga-20epoch-norotation/test.jsonl" \
    --output_dir "$output_dir" \
    --log_dir "$logs_dir" \
    \
    `# 新增并行化参数` \
    --num_envs $num_parallel_envs \
    \
    `# 模型超参数` \
    --input_dim 2 \
    --hidden_dim 128 \
    --max_seq_len 60 \
    \
    `# PPO 核心超参数 (n_steps 可以适当减小，因为总步数 = n_steps * num_envs)` \
    --total_timesteps 2000000 \
    --learning_rate 3e-4 \
    --n_steps 256 \
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