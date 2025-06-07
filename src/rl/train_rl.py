import numpy as np
import torch
import os
import sys
import json
import logging
import argparse
import random
from tqdm import tqdm
import wandb
import datetime
from typing import Dict, List

from environment import BinPackingEnvironment, BinPackingDataLoader
from dqn_agent import DQNAgent

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self, args):
        self.args = args
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 创建环境
        self.env = BinPackingEnvironment(
            max_parts=args.max_parts,
            state_dim=args.state_dim
        )
        
        # 创建智能体
        self.agent = DQNAgent(
            state_dim=args.state_dim,
            action_dim=args.max_parts,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            target_update_freq=args.target_update_freq,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            device=self.device
        )
        
        # 加载训练数据
        test_data_file = args.train_data_file.replace('train.jsonl', 'test.jsonl')
        self.data_loader = BinPackingDataLoader(args.train_data_file, test_data_file)
        logger.info(f"加载了 {len(self.data_loader)} 个训练实例")
        
        # 获取评估用的测试实例
        self.eval_instances = self.data_loader.get_test_instances(100)  # 获取前100个测试实例
        logger.info(f"准备了 {len(self.eval_instances)} 个测试实例用于评估")
        
        # 训练统计
        self.episode_rewards = []
        self.episode_efficiencies = []
        self.episode_losses = []
        
        # 创建保存目录
        os.makedirs(args.model_save_dir, exist_ok=True)
        
    def train_episode(self, instance: Dict = None) -> Dict:
        """训练一个episode"""
        # 重置环境

        if instance is None:
            instance = self.data_loader.get_random_instance()
        else:
            instance = self.data_loader.get_instance(instance)
        
        if instance is None:
            logger.warning("无法获取训练实例")
            return {}
            
        state = self.env.reset(instance)
        total_reward = 0
        episode_losses = []
        steps = 0
        
        while not self.env.done and steps < self.args.max_steps_per_episode:
            # 获取有效动作
            valid_actions = self.env.get_valid_actions()
            if not valid_actions:
                break
                
            # 选择动作
            action = self.agent.select_action(state, valid_actions)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 存储经验
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # 更新网络
            loss = self.agent.update()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
            
        # 记录统计信息
        final_efficiency = self.env._calculate_current_efficiency()
        metrics = self.env.get_placement_metrics()
        
        return {
            'reward': total_reward,
            'efficiency': final_efficiency,  # 现在是真正的材料利用率
            'material_utilization': metrics['material_utilization'],
            'bbox_ratio': metrics['bbox_ratio'],
            'compactness_score': metrics['compactness_score'],
            'steps': steps,
            'losses': episode_losses,
            'placed_parts': len(self.env.placed_parts),
            'valid_instance': True
        }
    
    def evaluate_agent(self, num_episodes: int = 50) -> Dict:
        """评估智能体性能"""
        self.agent.set_eval_mode()
        eval_rewards = []
        eval_efficiencies = []
        eval_material_utilizations = []
        eval_bbox_ratios = []
        eval_compactness_scores = []
        
        # 使用固定的测试实例进行评估
        eval_instances = self.eval_instances[:num_episodes] if len(self.eval_instances) >= num_episodes else self.eval_instances
        
        for instance in eval_instances:
            if instance is None:
                continue
                
            state = self.env.reset(instance)
            total_reward = 0
            steps = 0
            
            while not self.env.done and steps < self.args.max_steps_per_episode:
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break
                    
                # 贪婪策略评估
                action = self.agent.select_action(state, valid_actions)
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1
            
            # 获取详细指标
            final_efficiency = self.env._calculate_current_efficiency()
            metrics = self.env.get_placement_metrics()
            
            eval_rewards.append(total_reward)
            eval_efficiencies.append(final_efficiency)
            eval_material_utilizations.append(metrics['material_utilization'])
            eval_bbox_ratios.append(metrics['bbox_ratio'])
            eval_compactness_scores.append(metrics['compactness_score'])
        
        self.agent.set_train_mode()
        
        return {
            'avg_reward': np.mean(eval_rewards) if eval_rewards else 0,
            'avg_efficiency': np.mean(eval_efficiencies) if eval_efficiencies else 0,
            'avg_material_utilization': np.mean(eval_material_utilizations) if eval_material_utilizations else 0,
            'avg_bbox_ratio': np.mean(eval_bbox_ratios) if eval_bbox_ratios else 0,
            'avg_compactness_score': np.mean(eval_compactness_scores) if eval_compactness_scores else 0,
            'std_efficiency': np.std(eval_efficiencies) if eval_efficiencies else 0,
            'std_material_utilization': np.std(eval_material_utilizations) if eval_material_utilizations else 0,
            'episodes_evaluated': len(eval_rewards)
        }
    
    def train(self):
        """主训练循环"""
        logger.info("开始强化学习训练...")
        
        best_avg_efficiency = 0
        episodes_since_improvement = 0
        
        for episode in tqdm(range(self.args.num_episodes), desc="训练进度"):
            # 训练一个episode
            episode_result = self.train_episode(episode)
            
            if not episode_result.get('valid_instance', False):
                continue
                
            # 记录统计信息
            self.episode_rewards.append(episode_result['reward'])
            self.episode_efficiencies.append(episode_result['efficiency'])
            if episode_result['losses']:
                self.episode_losses.extend(episode_result['losses'])
            
            # 定期评估和记录
            if (episode + 1) % self.args.eval_freq == 0:
                eval_result = self.evaluate_agent(self.args.eval_episodes)
                
                # 获取智能体统计信息
                agent_stats = self.agent.get_stats()
                
                # 记录到wandb
                wandb_log = {
                    'episode': episode + 1,
                    'train_reward': episode_result['reward'],
                    'train_efficiency': episode_result['efficiency'],
                    'train_material_utilization': episode_result['material_utilization'],
                    'train_bbox_ratio': episode_result['bbox_ratio'],
                    'train_compactness_score': episode_result['compactness_score'],
                    'train_steps': episode_result['steps'],
                    'train_avg_loss': np.mean(episode_result['losses']) if episode_result['losses'] else 0,
                    'train_total_loss': np.sum(episode_result['losses']) if episode_result['losses'] else 0,
                    'eval_avg_reward': eval_result['avg_reward'],
                    'eval_avg_efficiency': eval_result['avg_efficiency'],
                    'eval_avg_material_utilization': eval_result['avg_material_utilization'],
                    'eval_avg_bbox_ratio': eval_result['avg_bbox_ratio'],
                    'eval_avg_compactness_score': eval_result['avg_compactness_score'],
                    'eval_std_efficiency': eval_result['std_efficiency'],
                    'eval_std_material_utilization': eval_result['std_material_utilization'],
                    'epsilon': agent_stats['epsilon'],
                    'avg_reward_last_100': agent_stats['avg_reward_last_100'],
                    'avg_loss_last_100': agent_stats['avg_loss_last_100'],
                    'total_steps': agent_stats['steps_done']
                }
                
                # 添加奖励统计信息（如果存在）
                if 'reward_mean' in agent_stats:
                    wandb_log.update({
                        'reward_mean': agent_stats['reward_mean'],
                        'reward_std': agent_stats['reward_std'],
                        'reward_count': agent_stats['reward_count']
                    })
                
                wandb.log(wandb_log)
                
                logger.info(
                    f"Episode {episode + 1}: "
                    f"Train Util={episode_result['material_utilization']:.3f}, "
                    f"Train BBox={episode_result['bbox_ratio']:.3f}, "
                    f"Train AvgLoss={np.mean(episode_result['losses']) if episode_result['losses'] else 0:.4f}, "
                    f"Eval Util={eval_result['avg_material_utilization']:.3f}±{eval_result['std_material_utilization']:.3f}, "
                    f"Eval BBox={eval_result['avg_bbox_ratio']:.3f}, "
                    f"Compact={eval_result['avg_compactness_score']:.3f}, "
                    f"ε={agent_stats['epsilon']:.3f}, "
                    f"Steps={agent_stats['steps_done']}"
                )
                
                # 保存最佳模型 - 使用材料利用率作为主要指标
                current_score = eval_result['avg_material_utilization']  # 材料利用率越高越好
                if current_score > best_avg_efficiency:
                    best_avg_efficiency = current_score
                    episodes_since_improvement = 0
                    
                    best_model_path = os.path.join(self.args.model_save_dir, 'best_rl_model.pth')
                    self.agent.save_model(best_model_path)
                    logger.info(f"保存新的最佳模型，材料利用率: {best_avg_efficiency:.3f}, "
                               f"边界框比例: {eval_result['avg_bbox_ratio']:.3f}, "
                               f"紧密度评分: {eval_result['avg_compactness_score']:.3f}")
                else:
                    episodes_since_improvement += self.args.eval_freq
                
                # 早停检查
                if episodes_since_improvement >= self.args.early_stop_patience:
                    logger.info(f"早停触发，在第 {episode + 1} 轮后停止训练")
                    break
            
            # 定期保存检查点
            if (episode + 1) % self.args.save_freq == 0:
                checkpoint_path = os.path.join(
                    self.args.model_save_dir, 
                    f'rl_checkpoint_episode_{episode + 1}.pth'
                )
                self.agent.save_model(checkpoint_path)
        
        # 保存最终模型
        final_model_path = os.path.join(self.args.model_save_dir, 'final_rl_model.pth')
        self.agent.save_model(final_model_path)
        logger.info(f"训练完成，最终模型已保存到: {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Deep Reinforcement Learning for Bin Packing")
    
    # 环境参数
    parser.add_argument("--max_parts", type=int, default=60, help="Maximum number of parts")
    parser.add_argument("--state_dim", type=int, default=256, help="State dimension")
    parser.add_argument("--max_steps_per_episode", type=int, default=100, help="Maximum steps per episode")
    
    # 智能体参数
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--epsilon_decay", type=int, default=50000, help="Epsilon decay steps")
    parser.add_argument("--target_update_freq", type=int, default=1000, help="Target network update frequency")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    
    # 训练参数
    parser.add_argument("--num_episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--eval_freq", type=int, default=100, help="Evaluation frequency")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--save_freq", type=int, default=1000, help="Model save frequency")
    parser.add_argument("--early_stop_patience", type=int, default=2000, help="Early stopping patience")
    
    # 数据和保存路径
    parser.add_argument("--train_data_file", type=str, 
                        default="./data/placement-0529-ga-20epoch-norotation/train.jsonl", 
                        help="Training data file")
    parser.add_argument("--model_save_dir", type=str, 
                        default="./output/rl_models", 
                        help="Model save directory")
    
    # WandB参数
    parser.add_argument("--wandb_project", type=str, 
                        default="bin-packing-rl", 
                        help="WandB project name")
    parser.add_argument("--wandb_name_prefix", type=str, 
                        default="dqn", 
                        help="WandB run name prefix")
    parser.add_argument("--wandb_mode", type=str, 
                        default="disabled", 
                        choices=["online", "offline", "disabled"], 
                        help="WandB mode")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 初始化wandb
    wandb.init(
        project=args.wandb_project,
        name=f"{args.wandb_name_prefix}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        settings=wandb.Settings(init_timeout=300),
        mode=args.wandb_mode,
        config=vars(args)
    )
    
    # 创建训练器并开始训练
    trainer = RLTrainer(args)
    trainer.train()
    
    wandb.finish()


if __name__ == '__main__':
    main() 