#!/usr/bin/env python3
"""
Debug脚本：展示从输入到输出reward到计算loss的完整训练流程
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入必要的模块
from src.environment.td_environment import TDDataLoader, TDEnvironment
from src.model.td_policy_network import TDPolicyNetwork
from src.reward.td_reward_calculator import TDRewardCalculator


def debug_training_flow():
    """调试训练流程"""
    
    print("=" * 80)
    print("开始调试训练流程")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    data_loader = TDDataLoader("data/placement-0529-ga-20epoch-norotation/train.jsonl")
    instance = data_loader.get_instance(0)
    
    print(f"实例数据结构:")
    print(f"  - parts数量: {len(instance['parts'])}")
    print(f"  - 实例keys: {list(instance.keys())}")
    print(f"  - bin_width: {instance.get('bin_width', instance.get('width', 'N/A'))}")
    print(f"  - bin_height: {instance.get('bin_height', instance.get('height', 'N/A'))}")
    print(f"  - 前3个parts: {instance['parts'][:3]}")
    
    # 2. 初始化模型和环境
    print("\n2. 初始化模型和环境...")
    policy_net = TDPolicyNetwork(input_dim=2, hidden_dim=128, seq_len=60)
    env = TDEnvironment()
    reward_calculator = TDRewardCalculator(w1=0.5, w2=0.0, w3=0.5)
    
    print(f"模型参数数量: {sum(p.numel() for p in policy_net.parameters())}")
    
    # 3. 重置环境并获取初始状态
    print("\n3. 重置环境...")
    state = env.reset(instance)
    num_parts = env.num_parts
    
    print(f"环境状态:")
    print(f"  - 零件数量: {num_parts}")
    print(f"  - 状态张量形状: {torch.FloatTensor(state).shape}")
    print(f"  - 前3个零件坐标: {state[:3]}")
    
    # 4. 准备输入数据
    print("\n4. 准备输入数据...")
    parts_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, num_parts, 2]
    target_sequence = torch.arange(num_parts).unsqueeze(0)  # [1, num_parts]
    valid_lengths = torch.tensor([num_parts])
    
    print(f"输入张量:")
    print(f"  - parts_tensor形状: {parts_tensor.shape}")
    print(f"  - target_sequence形状: {target_sequence.shape}")
    print(f"  - valid_lengths: {valid_lengths}")
    print(f"  - target_sequence内容: {target_sequence[0][:10]}...")
    
    # 5. 前向传播
    print("\n5. 模型前向传播...")
    policy_net.train()
    
    with torch.no_grad():
        # 编码器输出
        encoded_features = policy_net.encoder(parts_tensor)
        print(f"编码器输出形状: {encoded_features.shape}")
        
        # 解码器初始化
        batch_size = parts_tensor.size(0)
        hidden = policy_net.init_hidden(batch_size)
        print(f"初始隐藏状态形状: {[h.shape for h in hidden]}")
    
    # 进行完整的前向传播
    log_probs, action_sequence = policy_net(parts_tensor, target_sequence, valid_lengths)
    
    print(f"前向传播输出:")
    print(f"  - log_probs形状: {log_probs.shape}")
    print(f"  - action_sequence形状: {action_sequence.shape}")
    print(f"  - 前10个log_probs: {log_probs[0][:10]}")
    print(f"  - 前10个actions: {action_sequence[0][:10]}")
    
    # 6. 在环境中执行动作序列
    print("\n6. 在环境中执行动作...")
    env.reset(instance)  # 重置环境
    
    rewards = []
    step_info = []
    
    for step in range(num_parts):
        action = action_sequence[0, step].item()
        
        if env.is_action_valid(action):
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            
            step_info.append({
                'step': step,
                'action': action,
                'reward': reward,
                'done': done,
                'remaining_parts': len(env.remaining_parts)
            })
            
            if step < 5:  # 只打印前5步
                print(f"  步骤 {step}: 动作={action}, 奖励={reward:.4f}, 剩余零件={len(env.remaining_parts)}")
        else:
            print(f"  步骤 {step}: 无效动作={action}")
            rewards.append(0.0)
            break
    
    total_reward = sum(rewards)
    print(f"总奖励: {total_reward:.4f}")
    
    # 7. 获取最终指标
    print("\n7. 获取最终指标...")
    final_metrics = env.get_placement_metrics()
    print(f"最终指标:")
    for key, value in final_metrics.items():
        print(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")
    
    # 8. 计算损失
    print("\n8. 计算损失...")
    
    # 计算策略损失 (REINFORCE)
    policy_loss = 0.0
    baseline_reward = sum(rewards) / len(rewards) if rewards else 0.0
    
    for step in range(len(rewards)):
        if step < log_probs.size(1):
            advantage = rewards[step] - baseline_reward
            policy_loss -= log_probs[0, step] * advantage
    
    policy_loss = policy_loss / len(rewards) if rewards else torch.tensor(0.0)
    
    print(f"损失计算:")
    print(f"  - 基线奖励: {baseline_reward:.4f}")
    print(f"  - 策略损失: {policy_loss:.4f}")
    print(f"  - 奖励方差: {torch.tensor(rewards).var():.4f}" if rewards else "  - 奖励方差: 0.0")
    
    # 9. 反向传播
    print("\n9. 反向传播...")
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    
    optimizer.zero_grad()
    policy_loss.backward()
    
    # 计算梯度范数
    total_norm = 0
    param_count = 0
    for p in policy_net.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    total_norm = total_norm ** (1. / 2)
    
    print(f"梯度信息:")
    print(f"  - 梯度范数: {total_norm:.6f}")
    print(f"  - 有梯度的参数数量: {param_count}")
    
    optimizer.step()
    
    # 10. 总结
    print("\n" + "=" * 80)
    print("训练流程总结:")
    print("=" * 80)
    print(f"输入: {num_parts}个零件")
    print(f"模型输出: {action_sequence.shape[1]}个动作")
    print(f"执行步数: {len(rewards)}")
    print(f"总奖励: {total_reward:.4f}")
    print(f"平均奖励: {total_reward/len(rewards):.4f}" if rewards else "平均奖励: 0.0")
    print(f"材料利用率: {final_metrics.get('material_utilization', 0):.3f}")
    print(f"策略损失: {policy_loss:.4f}")
    print(f"梯度范数: {total_norm:.6f}")
    print("=" * 80)
    
    return {
        'instance': instance,
        'num_parts': num_parts,
        'action_sequence': action_sequence[0].tolist(),
        'rewards': rewards,
        'total_reward': total_reward,
        'final_metrics': final_metrics,
        'policy_loss': policy_loss.item(),
        'gradient_norm': total_norm
    }


if __name__ == "__main__":
    try:
        result = debug_training_flow()
        print(f"\n调试完成！结果已保存。")
    except Exception as e:
        print(f"调试过程中出错: {e}")
        import traceback
        traceback.print_exc() 