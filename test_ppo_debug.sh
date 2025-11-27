#!/bin/bash
# PPO 快速测试脚本 - 用于查看debug输出

# --- 配置 ---
timestamp=$(date +%Y%m%d_%H%M%S)
output_dir="./models/ppo_test_${timestamp}"
logs_dir="./logs/ppo_test_${timestamp}"

# 创建目录
mkdir -p "$output_dir" "$logs_dir"

echo "=== 快速测试模式 ==="
echo "模型将保存到: $output_dir"
echo "日志将保存到: $logs_dir"
echo ""

# 执行 PPO 训练脚本（使用较少的步数以便快速查看输出）
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
    `# PPO 核心超参数 - 减少步数以快速测试` \
    --total_timesteps 1000 \
    --learning_rate 3e-4 \
    --n_steps 128 \
    --batch_size 32 \
    --n_epochs 2 \
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
    --eval_interval 500 \
    --max_test_instances 10 \
    --use_wandb \
    --wandb_project "ppo_svgnest_test" \
    --wandb_run_name "ppo_test_${timestamp}"

echo ""
echo "测试完成。请查看日志文件: $logs_dir/train_ppo.log"

