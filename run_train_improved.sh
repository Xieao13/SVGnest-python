#!/bin/bash

# --- Configuration Parameters ---

# Training Parameters
LEARNING_RATE=0.001
BATCH_SIZE=64
HIDDEN_DIM=256
EPOCHS=10
MAX_SEQ_LEN=60
WEIGHT_DECAY=0.01

# Data and Model Paths
TRAIN_DATA_FILE="./data/placement-0529-ga-20epoch-norotation/train.jsonl"
TEST_DATA_FILE="./data/placement-0529-ga-20epoch-norotation/test.jsonl"
BEST_MODEL_DIR="./output/models"
FINAL_MODEL_DIR="./output/models"
mkdir -p $FINAL_MODEL_DIR

# WandB Parameters
WANDB_PROJECT="bin-packing-placement-improved"
WANDB_NAME_PREFIX="improved-placement"
# WANDB_MODE="online"


python src/train_placement_improved.py \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --epochs $EPOCHS \
    --max_seq_len $MAX_SEQ_LEN \
    --weight_decay $WEIGHT_DECAY \
    --train_data_file $TRAIN_DATA_FILE \
    --test_data_file $TEST_DATA_FILE > ./log/train_placement_improved.log 2>&1