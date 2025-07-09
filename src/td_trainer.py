import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import json
import os
from typing import List, Dict, Tuple
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.td_policy_network import TDPolicyNetwork
from src.reward.td_reward_calculator import TDRewardCalculator
from src.environment.td_environment import TDEnvironment, TDDataLoader

logger = logging.getLogger(__name__)


class TDTrainer:
    """时序差分训练器 - 使用训练数据直接训练"""
    
    def __init__(self, 
                 input_dim=2,
                 hidden_dim=128,
                 max_seq_len=60,
                 learning_rate=1e-4,
                 reward_weights=(0.7, 0.0, 0.3)):
        """
        初始化训练器
        
        Args:
            input_dim: 输入维度（零件特征维度）
            hidden_dim: 隐藏层维度
            max_seq_len: 最大序列长度
            learning_rate: 学习率
            reward_weights: 奖励权重
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 创建策略网络
        self.policy_net = TDPolicyNetwork(input_dim, hidden_dim, max_seq_len).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # 创建环境
        self.env = TDEnvironment(max_parts=max_seq_len, reward_weights=reward_weights)
        
        # 训练统计
        self.train_stats = {
            'episodes': 0,
            'total_loss': 0,
            'total_reward': 0,
            'losses': [],
            'rewards': [],
            'utilization_rates': []
        }
        
        logger.info("TDTrainer初始化完成")

    def train_on_instance(self, instance: Dict) -> Dict:
        """在单个实例上训练"""
        # 重置环境
        state = self.env.reset(instance)
        
        # 转换为tensor
        parts_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, seq_len, input_dim]
        
        # 使用模型预测动作序列（让模型学习策略）
        actual_parts = min(self.env.num_parts, self.policy_net.seq_len)
        
        # 使用模型预测（不使用teacher forcing）
        with torch.no_grad():
            predicted_sequence = self.policy_net.predict(parts_tensor, 
                                                       valid_lengths=torch.tensor([actual_parts]))
        
        # 在环境中执行预测的序列，收集奖励
        total_reward = 0
        step_rewards = []
        action_sequence = []
        
        # 重置环境以执行预测的动作
        self.env.reset(instance)
        
        for step in range(actual_parts):
            action = predicted_sequence[0, step].item()
            
            # 检查动作是否有效
            if not self.env.is_action_valid(action):
                # 如果动作无效，给予负奖励并跳过
                reward = -1.0
                step_rewards.append(reward)
                total_reward += reward
                continue
            
            # 执行动作获得奖励
            _, reward, done, info = self.env.step(action)
            total_reward += reward
            step_rewards.append(reward)
            action_sequence.append(action)
            
            if done:
                break
        
        # 现在使用收集到的奖励来训练模型
        # 重新进行前向传播以计算损失
        target_sequence = torch.LongTensor([action_sequence]).to(self.device)
        
        if len(action_sequence) > 0:
            logits = self.policy_net(parts_tensor, target_sequence, 
                                   valid_lengths=torch.tensor([len(action_sequence)]))
            
            # 计算策略梯度损失
            total_loss = 0
            step_losses = []
            
            for step in range(len(action_sequence)):
                action = action_sequence[step]
                reward = step_rewards[step]
                
                # 计算策略损失
                step_logits = logits[0, step, :actual_parts]  # 只取有效的零件数量
                step_log_probs = torch.log_softmax(step_logits, dim=0)
                action_log_prob = step_log_probs[action]
                
                # 策略梯度损失：-log_prob * reward
                step_loss = -action_log_prob * reward
                total_loss += step_loss
                step_losses.append(step_loss.item())
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        else:
            total_loss = torch.tensor(0.0)
            step_losses = []
        
        # 获取最终指标
        final_metrics = self.env.get_placement_metrics()
        
        return {
            'loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'reward': total_reward,
            'step_losses': step_losses,
            'step_rewards': step_rewards,
            'final_metrics': final_metrics,
            'num_steps': len(step_rewards)
        }

    def train_epoch(self, data_loader: TDDataLoader, max_instances: int = None) -> Dict:
        """训练一个epoch"""
        self.policy_net.train()
        
        epoch_stats = {
            'total_loss': 0,
            'total_reward': 0,
            'num_instances': 0,
            'avg_utilization': 0,
            'instance_stats': []
        }
        
        # 限制训练实例数量
        num_instances = min(len(data_loader), max_instances) if max_instances else len(data_loader)
        
        for i in range(num_instances):
            instance = data_loader.get_instance(i)
            if instance is None:
                continue
                
            try:
                # 训练单个实例
                result = self.train_on_instance(instance)
                
                # 更新统计
                epoch_stats['total_loss'] += result['loss']
                epoch_stats['total_reward'] += result['reward']
                epoch_stats['num_instances'] += 1
                
                # 记录材料利用率
                utilization = result['final_metrics'].get('material_utilization', 0)
                epoch_stats['avg_utilization'] += utilization
                
                # 记录实例统计
                epoch_stats['instance_stats'].append({
                    'instance_id': i,
                    'loss': result['loss'],
                    'reward': result['reward'],
                    'utilization': utilization,
                    'num_steps': result['num_steps']
                })
                
                # 每100个实例打印一次进度
                if (i + 1) % 100 == 0:
                    avg_loss = epoch_stats['total_loss'] / epoch_stats['num_instances']
                    avg_reward = epoch_stats['total_reward'] / epoch_stats['num_instances']
                    avg_util = epoch_stats['avg_utilization'] / epoch_stats['num_instances']
                    
                    logger.info(f"进度 {i+1}/{num_instances}: "
                              f"平均损失={avg_loss:.4f}, "
                              f"平均奖励={avg_reward:.4f}, "
                              f"平均利用率={avg_util:.3f}")
                
            except Exception as e:
                logger.warning(f"训练实例 {i} 时出错: {e}")
                continue
        
        # 计算平均值
        if epoch_stats['num_instances'] > 0:
            epoch_stats['avg_loss'] = epoch_stats['total_loss'] / epoch_stats['num_instances']
            epoch_stats['avg_reward'] = epoch_stats['total_reward'] / epoch_stats['num_instances']
            epoch_stats['avg_utilization'] = epoch_stats['avg_utilization'] / epoch_stats['num_instances']
        else:
            # 如果没有成功的实例，设置默认值
            epoch_stats['avg_loss'] = 0.0
            epoch_stats['avg_reward'] = 0.0
            epoch_stats['avg_utilization'] = 0.0
        
        return epoch_stats

    def evaluate(self, data_loader: TDDataLoader, num_instances: int = 100) -> Dict:
        """评估模型"""
        self.policy_net.eval()
        
        eval_stats = {
            'total_reward': 0,
            'avg_utilization': 0,
            'num_instances': 0,
            'results': []
        }
        
        with torch.no_grad():
            for i in range(min(num_instances, len(data_loader))):
                instance = data_loader.get_test_instance(i) if hasattr(data_loader, 'get_test_instance') else data_loader.get_instance(i)
                if instance is None:
                    continue
                    
                try:
                    # 重置环境
                    state = self.env.reset(instance)
                    parts_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    
                    # 使用贪心策略预测
                    predictions = self.policy_net.predict(parts_tensor, 
                                                        valid_lengths=torch.tensor([self.env.num_parts]))
                    
                    # 在环境中执行预测的序列
                    total_reward = 0
                    for step in range(self.env.num_parts):
                        action = predictions[0, step].item()
                        if self.env.is_action_valid(action):
                            _, reward, done, _ = self.env.step(action)
                            total_reward += reward
                            if done:
                                break
                    
                    # 获取最终指标
                    final_metrics = self.env.get_placement_metrics()
                    utilization = final_metrics.get('material_utilization', 0)
                    
                    eval_stats['total_reward'] += total_reward
                    eval_stats['avg_utilization'] += utilization
                    eval_stats['num_instances'] += 1
                    
                    eval_stats['results'].append({
                        'instance_id': i,
                        'reward': total_reward,
                        'utilization': utilization,
                        'final_metrics': final_metrics
                    })
                    
                except Exception as e:
                    logger.warning(f"评估实例 {i} 时出错: {e}")
                    continue
        
        # 计算平均值
        if eval_stats['num_instances'] > 0:
            eval_stats['avg_reward'] = eval_stats['total_reward'] / eval_stats['num_instances']
            eval_stats['avg_utilization'] = eval_stats['avg_utilization'] / eval_stats['num_instances']
        
        return eval_stats

    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_stats': self.train_stats
        }, filepath)
        logger.info(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_stats = checkpoint.get('train_stats', self.train_stats)
        logger.info(f"模型已从 {filepath} 加载")


def main():
    """主训练函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 检查数据文件
    train_file = "./data/placement-0529-ga-20epoch-norotation/train.jsonl"
    if not os.path.exists(train_file):
        logger.error(f"训练数据文件不存在: {train_file}")
        return
    
    # 创建数据加载器
    data_loader = TDDataLoader(train_file)
    logger.info(f"加载了 {len(data_loader)} 个训练实例")
    
    # 创建训练器
    trainer = TDTrainer(
        input_dim=2,
        hidden_dim=128,
        max_seq_len=60,
        learning_rate=1e-4,
        reward_weights=(0.4, 0.3, 0.4)
    )
    
    # 训练
    num_epochs = 5
    for epoch in range(num_epochs):
        logger.info(f"开始第 {epoch + 1} 轮训练...")
        
        # 训练一个epoch（使用更多实例）
        epoch_stats = trainer.train_epoch(data_loader, max_instances=10000)
        
        logger.info(f"第 {epoch + 1} 轮训练完成:")
        logger.info(f"  平均损失: {epoch_stats['avg_loss']:.4f}")
        logger.info(f"  平均奖励: {epoch_stats['avg_reward']:.4f}")
        logger.info(f"  平均利用率: {epoch_stats['avg_utilization']:.3f}")
        logger.info(f"  训练实例数: {epoch_stats['num_instances']}")
        
        # 保存模型
        model_path = f"./models/td_policy_epoch_{epoch + 1}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_model(model_path)
    
    logger.info("训练完成!")


if __name__ == '__main__':
    main() 