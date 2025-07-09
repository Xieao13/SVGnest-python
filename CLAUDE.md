# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project for **SVG nesting/bin packing optimization** using deep learning approaches. The project combines evolutionary algorithms (genetic algorithm) with neural networks to solve the 2D bin packing problem for SVG shapes.

### Key Technologies
- **Deep Learning**: PyTorch-based neural networks (Pointer Networks, Transformers)
- **Reinforcement Learning**: DQN (Deep Q-Networks) for sequential decision making
- **Evolutionary Algorithms**: Genetic Algorithm for generating training data
- **Computer Vision**: SVG parsing and geometric processing

## Architecture Overview

### Core Components

1. **Data Generation** (`svgnest_*.py`)
   - `svgnest_ga.py`: Genetic algorithm for generating optimal placements
   - `svgnest_random.py`: Random placement baseline
   - `svgnest_v0.py`: Base SVG nesting implementation

2. **Neural Network Models** (`src/model/`)
   - `placement_only_model.py`: Pointer network for placement sequence prediction
   - `td_policy_network.py`: Temporal difference policy network
   - `model_v1.py`: Earlier model versions

3. **Training Systems**
   - **Supervised Learning**: `src/train_improved.py`, `src/train_placement_improved.py`
   - **Reinforcement Learning**: `src/rl/train_rl.py`, `src/rl/dqn_agent.py`
   - **Temporal Difference**: `src/td_trainer.py`

4. **Environment & Rewards** (`src/environment/`, `src/reward/`)
   - `td_environment.py`: Environment for temporal difference learning
   - `td_reward_calculator.py`: Multi-dimensional reward calculation

5. **Utilities** (`utils/`)
   - `geometry_util.py`: Geometric computations
   - `svg_parser.py`: SVG file parsing
   - `placement_worker.py`: Core placement engine

### Data Flow

1. **Data Generation**: Genetic algorithm generates high-quality placement sequences
2. **Training Data**: Generated placements stored in `data/` directory as JSONL files
3. **Model Training**: Various neural networks trained on generated data
4. **Evaluation**: Models evaluated on test sets and compared with baselines

## Common Development Commands

### Training Scripts

```bash
# Supervised learning with improved pointer network
./run_train_improved.sh

# Reinforcement learning with DQN
./run_train_rl.sh

# Alternative RL training approach
./run_train_rl_v1.sh

# Direct Python training calls
python src/train_improved.py
python src/train_placement_improved.py
python src/rl/train_rl.py
python src/td_trainer.py
```

### Testing & Evaluation

```bash
# Test data loading
python test_data_loading.py

# Test training pipeline
python test_training_fix.py

# Debug training flow
python src/debug_training_flow.py

# Inference with trained models
python predict_placement_single.py
python predict_placement_file.py
```

### Data Processing

```bash
# Create training dataset
python src/preprocess/create_dataset.py

# Split data into train/test
python src/preprocess/data_split.py
```

## Key Configuration

### Model Parameters
- **Max sequence length**: 60 parts
- **Hidden dimensions**: 128-512 (model dependent)
- **Learning rates**: 0.001-0.003
- **Batch sizes**: 32-64

### Training Data
- **Location**: `data/placement-0529-ga-20epoch-norotation/`
- **Format**: JSONL with part sequences and placements
- **Size**: ~100k samples with 0.72-0.85 efficiency

### Output Directories
- **Models**: `output/models/`, `output/rl_models/`
- **Logs**: `log/`
- **Results**: `output/`

## Architecture Patterns

### Pointer Network Design
- **Encoder**: Transformer-based part feature encoding
- **Decoder**: Autoregressive pointer mechanism
- **Attention**: Multi-head attention for part selection

### Reward Function Components
1. **Compactness**: Ratio of part areas to bounding box area
2. **Fitting**: Contact length with bin boundaries and other parts
3. **Utilization**: Waste area minimization

### Training Approaches
1. **Supervised**: Learning from GA-generated sequences
2. **Reinforcement**: DQN with environment interaction
3. **Temporal Difference**: Step-by-step policy optimization

## Development Notes

### Model Training Tips
- Use teacher forcing ratio decay for autoregressive models
- Implement gradient clipping for training stability
- Monitor both loss and placement efficiency metrics

### Data Requirements
- Ensure sufficient training data diversity (different bin sizes, part combinations)
- Validate data quality with efficiency metrics
- Use proper train/validation/test splits

### Performance Optimization
- Utilize GPU acceleration for neural network training
- Implement efficient geometric computations
- Cache expensive placement calculations

## Research Context

This project implements the research approach of "using genetic algorithms to generate training data for pointer networks in bin packing problems." The core innovation is combining evolutionary optimization with deep learning for geometric optimization tasks.

The temporal difference approach (详细设计见 `时序差分纯策略学习方案.md`) represents an advanced training strategy for immediate reward feedback in sequential decision making.