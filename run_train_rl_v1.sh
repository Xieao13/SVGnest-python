#!/bin/bash

# --- Deep Reinforcement Learning Training Script for Bin Packing ---
# Environment Parameters
MAX_PARTS=60
STATE_DIM=256
MAX_STEPS_PER_EPISODE=100

# Agent Parameters
LEARNING_RATE=0.001
GAMMA=0.99
EPSILON_START=1.0
EPSILON_END=0.05
EPSILON_DECAY=20000
TARGET_UPDATE_FREQ=1000
BUFFER_SIZE=100000
BATCH_SIZE=32
HIDDEN_DIM=512

# Training Parameters
NUM_EPISODES=50000
EVAL_FREQ=100
EVAL_EPISODES=100
SAVE_FREQ=2000
EARLY_STOP_PATIENCE=2000

# Data and Model Paths
TRAIN_DATA_FILE="./data/placement-0529-ga-20epoch-norotation/train.jsonl"
MODEL_SAVE_DIR="./output/rl_models_v1"
mkdir -p $MODEL_SAVE_DIR

# Create log directory
LOG_DIR="./log"
mkdir -p $LOG_DIR

# WandB Parameters
WANDB_PROJECT="bin-packing-rl-v1"
WANDB_NAME_PREFIX="dqn"
WANDB_MODE="online"  # Change to "online" to enable WandB logging

echo "Starting Deep Reinforcement Learning Training for Bin Packing..."
echo "Training data: $TRAIN_DATA_FILE"
echo "Model save directory: $MODEL_SAVE_DIR"
echo "Log directory: $LOG_DIR"

python ./src/rl/train_rl.py \
    --max_parts $MAX_PARTS \
    --state_dim $STATE_DIM \
    --max_steps_per_episode $MAX_STEPS_PER_EPISODE \
    --learning_rate $LEARNING_RATE \
    --gamma $GAMMA \
    --epsilon_start $EPSILON_START \
    --epsilon_end $EPSILON_END \
    --epsilon_decay $EPSILON_DECAY \
    --target_update_freq $TARGET_UPDATE_FREQ \
    --buffer_size $BUFFER_SIZE \
    --batch_size $BATCH_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --num_episodes $NUM_EPISODES \
    --eval_freq $EVAL_FREQ \
    --eval_episodes $EVAL_EPISODES \
    --save_freq $SAVE_FREQ \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --train_data_file $TRAIN_DATA_FILE \
    --model_save_dir $MODEL_SAVE_DIR \
    --wandb_project $WANDB_PROJECT \
    --wandb_name_prefix $WANDB_NAME_PREFIX \
    --wandb_mode $WANDB_MODE  > $LOG_DIR/train_rl_v1.log 2>&1

echo "Training completed. Check $LOG_DIR/train_rl_v1.log for details."
echo "Models saved in: $MODEL_SAVE_DIR" 