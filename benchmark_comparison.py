import torch
import numpy as np
import argparse
import os
import sys
import json
import logging
import time
import random
from typing import Dict, List
from tqdm import tqdm

# 移除 pandas 依赖以解决 numpy 版本冲突
# import pandas as pd

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.model.ppo_actor_critic import PPOActorCritic
from src.environment.td_environment import TDEnvironment
from utils.placement_worker import PlacementWorker
from utils.geometry_util import GeometryUtil

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_placement(bin_polygon, parts, placement_sequence):
    """执行具体的放置计算，返回利用率"""
    # 构建顺序
    ordered_parts = [parts[i] for i in placement_sequence]
    rotations = [0] * len(ordered_parts)
    
    config = {
        'spacing': 0,
        'clipperScale': 10000000,
        'curveTolerance': 0
    }
    
    try:
        worker = PlacementWorker(
            bin_polygon=bin_polygon,
            paths=ordered_parts,
            ids=placement_sequence,
            rotations=rotations,
            config=config
        )
        placed, unplaced, bounding_box_area = worker.place_paths()
        
        # 计算利用率
        placed_area = sum(abs(GeometryUtil.polygon_area(p)) for p in placed)
        if bounding_box_area > 0:
            return placed_area / bounding_box_area
        return 0.0
    except Exception as e:
        return 0.0

def evaluate_random(env, instance, num_repeats=1):
    """随机策略：随机打乱多次取最好"""
    best_util = 0.0
    
    # 这里只跑一次随机，模拟纯随机基线
    # 如果想要 strong baseline，可以 num_repeats=10
    for _ in range(num_repeats):
        seq = list(range(len(instance['parts'])))
        random.shuffle(seq)
        # 截断到 env.num_parts (如果 instance parts 更多)
        seq = seq[:env.num_parts]
        
        util = run_placement(env.bin_polygon, env.parts, seq)
        best_util = max(best_util, util)
        
    return best_util

def evaluate_heuristic(env, instance):
    """启发式策略：按面积降序排列 (Area Descending)"""
    # 计算每个零件面积
    areas = []
    for i, part in enumerate(env.parts):
        area = abs(GeometryUtil.polygon_area(part))
        areas.append((i, area))
    
    # 按面积降序
    areas.sort(key=lambda x: x[1], reverse=True)
    seq = [x[0] for x in areas]
    
    util = run_placement(env.bin_polygon, env.parts, seq)
    return util

def evaluate_ppo(env, policy, device, max_seq_len):
    """PPO 模型策略"""
    state_np = env._get_state() # 获取当前状态 (只需零件尺寸特征)
    
    with torch.no_grad():
        state = torch.tensor(state_np, dtype=torch.float32, device=device)
        encoder_outputs = policy.encode_inputs(state.unsqueeze(0)).squeeze(0)
        decoder_input, decoder_hidden = policy.get_initial_state(1, device)
        decoder_input = decoder_input.squeeze(0)
        selected_mask = torch.zeros(max_seq_len, dtype=torch.bool, device=device)
        valid_length = torch.tensor([env.num_parts], device=device)

        actions = []
        done = False
        
        # 模拟环境交互生成序列
        for _ in range(env.num_parts):
            logits, _, next_decoder_hidden = policy.forward(
                encoder_outputs.unsqueeze(0), decoder_input.unsqueeze(0),
                decoder_hidden, selected_mask.unsqueeze(0), valid_length
            )
            action = torch.argmax(logits, dim=1)
            action_item = action.item()
            
            actions.append(action_item)
            
            decoder_hidden = next_decoder_hidden
            decoder_input = encoder_outputs[action_item]
            selected_mask[action_item] = True
            
    # 使用生成的序列进行评估
    util = run_placement(env.bin_polygon, env.parts, actions)
    return util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models_ppo/ppo_policy_best.pth')
    parser.add_argument('--test_file', type=str, default='./data/placement-0529-ga-20epoch-norotation/test.jsonl')
    parser.add_argument('--max_seq_len', type=int, default=60)
    parser.add_argument('--limit', type=int, default=100, help='Number of instances to test (0 for all)')
    args = parser.parse_args()

    print(f"Loading PPO model from {args.model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载 PPO 模型
    if os.path.exists(args.model_path):
        # PyTorch 2.6+ fix: weights_only=False needed for loading argparse.Namespace
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model_args = checkpoint.get('args')
        input_dim = getattr(model_args, 'input_dim', 2)
        hidden_dim = getattr(model_args, 'hidden_dim', 128)
        policy = PPOActorCritic(input_dim, hidden_dim, args.max_seq_len).to(device)
        policy.load_state_dict(checkpoint['model_state_dict'])
        policy.eval()
    else:
        print("Error: Model path not found!")
        return

    # 加载测试数据
    print(f"Loading test data from {args.test_file}...")
    instances = []
    with open(args.test_file, 'r') as f:
        for line in f:
            instances.append(json.loads(line))
    
    if args.limit > 0:
        instances = instances[:args.limit]
    
    print(f"Starting comparison on {len(instances)} instances...")
    print("-" * 60)
    print(f"{'ID':<5} | {'Random':<10} | {'AreaDesc':<10} | {'PPO':<10} | {'Best':<10}")
    print("-" * 60)

    results = []
    env = TDEnvironment(max_parts=args.max_seq_len)

    for i, instance in enumerate(tqdm(instances, desc="Running Tests")):
        # 重置环境以加载实例
        env.reset(instance)
        
        # 1. Random
        util_random = evaluate_random(env, instance)
        
        # 2. Heuristic (Area Descending) - 作为简单 GA 的替代/基准
        util_heuristic = evaluate_heuristic(env, instance)
        
        # 3. PPO
        util_ppo = evaluate_ppo(env, policy, device, args.max_seq_len)
        
        # 记录
        best_method = "Random"
        max_val = util_random
        
        if util_heuristic > max_val:
            max_val = util_heuristic
            best_method = "AreaDesc"
        if util_ppo > max_val:
            max_val = util_ppo
            best_method = "PPO"
            
        results.append({
            'id': i,
            'Random': util_random,
            'AreaDesc': util_heuristic,
            'PPO': util_ppo,
            'Winner': best_method
        })
        
        # 实时打印前 10 个和每 50 个的结果
        if i < 10 or (i+1) % 50 == 0:
            # 避免 tqdm 冲突，使用 tqdm.write
            tqdm.write(f"{i:<5} | {util_random:.4f}     | {util_heuristic:.4f}     | {util_ppo:.4f}     | {best_method}")

    # 汇总统计
    # 使用纯 Python 替代 pandas 统计和打印
    print("\n" + "=" * 60)
    print("Final Benchmark Results Summary")
    print("=" * 60)
    
    methods = ['Random', 'AreaDesc', 'PPO']
    stats = {}
    
    # 计算统计数据
    for method in methods:
        values = [r[method] for r in results]
        stats[method] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # 打印统计表格
    print(f"{'Metric':<10} | {'Random':<12} | {'AreaDesc':<12} | {'PPO':<12}")
    print("-" * 60)
    for metric in ['mean', 'std', 'min', 'max']:
        print(f"{metric:<10} | {stats['Random'][metric]:<12.4f} | {stats['AreaDesc'][metric]:<12.4f} | {stats['PPO'][metric]:<12.4f}")
    
    print("\nWin Rates:")
    win_counts = {}
    for r in results:
        winner = r['Winner']
        win_counts[winner] = win_counts.get(winner, 0) + 1
        
    for method in methods:
        count = win_counts.get(method, 0)
        print(f"{method}: {count}/{len(instances)} ({count/len(instances)*100:.1f}%)")

    # 提升幅度分析
    random_vals = np.array([r['Random'] for r in results])
    areadesc_vals = np.array([r['AreaDesc'] for r in results])
    ppo_vals = np.array([r['PPO'] for r in results])
    
    # 避免除以零
    random_vals = np.maximum(random_vals, 1e-6)
    areadesc_vals = np.maximum(areadesc_vals, 1e-6)
    
    ppo_improvement = (ppo_vals - random_vals) / random_vals * 100
    ppo_vs_heuristic = (ppo_vals - areadesc_vals) / areadesc_vals * 100
    
    print(f"\nAverage Improvement over Random: {np.mean(ppo_improvement):.2f}%")
    print(f"Average Improvement over AreaDesc: {np.mean(ppo_vs_heuristic):.2f}%")

if __name__ == '__main__':
    main()

