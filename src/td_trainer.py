import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import json
import os
import argparse
import time
from typing import List, Dict, Tuple
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.td_policy_network import TDPolicyNetwork
from src.reward.td_reward_calculator import TDRewardCalculator
from src.environment.td_environment import TDEnvironment, TDDataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

logger = logging.getLogger(__name__)


class TDTrainer:
    """时序差分训练器 - 使用训练数据直接训练"""
    
    def __init__(self, 
                 input_dim=2,
                 hidden_dim=128,
                 max_seq_len=60,
                 learning_rate=3e-4,
                 reward_weights=(0.4, 0.3, 0.3),
                 use_baseline=True,
                 clip_grad_norm=1.0,
                 use_wandb=True):
        """
        初始化训练器
        
        Args:
            input_dim: 输入维度（零件特征维度）
            hidden_dim: 隐藏层维度
            max_seq_len: 最大序列长度
            learning_rate: 学习率
            reward_weights: 奖励权重
            use_baseline: 是否使用baseline减少方差
            clip_grad_norm: 梯度裁剪的最大范数
            use_wandb: 是否使用wandb进行监控
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.use_baseline = use_baseline
        self.clip_grad_norm = clip_grad_norm
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        logger.info(f"使用设备: {self.device}")
        
        # 创建策略网络
        self.policy_net = TDPolicyNetwork(input_dim, hidden_dim, max_seq_len).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # 创建环境
        self.env = TDEnvironment(max_parts=max_seq_len, reward_weights=reward_weights)
        
        # 基线值（用于减少方差）
        self.baseline = 0.0
        self.baseline_momentum = 0.9
        
        # 训练统计
        self.train_stats = {
            'episodes': 0,
            'total_loss': 0,
            'total_reward': 0,
            'losses': [],
            'rewards': [],
            'utilization_rates': [],
            'best_utilization': 0.0,
            'best_reward': float('-inf')
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
                
                # 策略梯度损失：-log_prob * (reward - baseline)
                if self.use_baseline:
                    advantage = reward - self.baseline
                else:
                    advantage = reward
                    
                step_loss = -action_log_prob * advantage
                total_loss += step_loss
                step_losses.append(step_loss.item())
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.clip_grad_norm)
            
            # 更新baseline
            if self.use_baseline and len(step_rewards) > 0:
                current_return = sum(step_rewards) / len(step_rewards)
                self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * current_return
            
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
            'num_steps': len(step_rewards),
            'baseline': self.baseline
        }

    def train_epoch(self, data_loader: TDDataLoader, epoch: int, max_instances: int = None) -> Dict:
        """训练一个epoch"""
        self.policy_net.train()
        
        epoch_stats = {
            'total_loss': 0,
            'total_reward': 0,
            'num_instances': 0,
            'avg_utilization': 0,
            'instance_stats': [],
            'avg_baseline': 0
        }
        
        # 限制训练实例数量
        num_instances = min(len(data_loader), max_instances) if max_instances else len(data_loader)
        
        start_time = time.time()
        
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
                epoch_stats['avg_baseline'] += result['baseline']
                
                # 记录材料利用率
                utilization = result['final_metrics'].get('material_utilization', 0)
                epoch_stats['avg_utilization'] += utilization
                
                # 更新最佳记录
                if utilization > self.train_stats['best_utilization']:
                    self.train_stats['best_utilization'] = utilization
                if result['reward'] > self.train_stats['best_reward']:
                    self.train_stats['best_reward'] = result['reward']
                
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
                    avg_baseline = epoch_stats['avg_baseline'] / epoch_stats['num_instances']
                    
                    logger.info(f"Epoch {epoch+1} - 进度 {i+1}/{num_instances}: "
                              f"平均损失={avg_loss:.4f}, "
                              f"平均奖励={avg_reward:.4f}, "
                              f"平均利用率={avg_util:.3f}, "
                              f"基线值={avg_baseline:.4f}")
                    
                    # 记录到wandb
                    if self.use_wandb:
                        wandb.log({
                            'train/step_loss': avg_loss,
                            'train/step_reward': avg_reward,
                            'train/step_utilization': avg_util,
                            'train/baseline': avg_baseline,
                            'train/progress': (i + 1) / num_instances
                        })
                
            except Exception as e:
                logger.warning(f"训练实例 {i} 时出错: {e}")
                continue
        
        # 计算平均值
        if epoch_stats['num_instances'] > 0:
            epoch_stats['avg_loss'] = epoch_stats['total_loss'] / epoch_stats['num_instances']
            epoch_stats['avg_reward'] = epoch_stats['total_reward'] / epoch_stats['num_instances']
            epoch_stats['avg_utilization'] = epoch_stats['avg_utilization'] / epoch_stats['num_instances']
            epoch_stats['avg_baseline'] = epoch_stats['avg_baseline'] / epoch_stats['num_instances']
        else:
            # 如果没有成功的实例，设置默认值
            epoch_stats['avg_loss'] = 0.0
            epoch_stats['avg_reward'] = 0.0
            epoch_stats['avg_utilization'] = 0.0
            epoch_stats['avg_baseline'] = 0.0
        
        epoch_stats['epoch_time'] = time.time() - start_time
        
        return epoch_stats

    def evaluate(self, data_loader: TDDataLoader, num_instances: int = 100) -> Dict:
        """评估模型"""
        self.policy_net.eval()
        
        eval_stats = {
            'total_reward': 0,
            'avg_utilization': 0,
            'num_instances': 0,
            'results': [],
            'best_utilization': 0.0,
            'worst_utilization': 1.0
        }
        
        start_time = time.time()
        
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
                    
                    # 更新最佳和最差记录
                    if utilization > eval_stats['best_utilization']:
                        eval_stats['best_utilization'] = utilization
                    if utilization < eval_stats['worst_utilization']:
                        eval_stats['worst_utilization'] = utilization
                    
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
        
        eval_stats['eval_time'] = time.time() - start_time
        
        return eval_stats

    def save_model(self, filepath: str, epoch: int = None, is_best: bool = False):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_stats': self.train_stats,
            'baseline': self.baseline,
            'epoch': epoch,
            'learning_rate': self.learning_rate
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"模型已保存到: {filepath}")
        
        # 如果是最佳模型，也保存一个best.pth的副本
        if is_best:
            best_path = filepath.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"最佳模型已保存到: {best_path}")

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_stats = checkpoint.get('train_stats', self.train_stats)
        self.baseline = checkpoint.get('baseline', self.baseline)
        logger.info(f"模型已从 {filepath} 加载")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='时序差分策略学习训练')
    
    # 数据相关参数
    parser.add_argument('--train_file', type=str, default='./data/placement-0529-ga-20epoch-norotation/train.jsonl',
                        help='训练数据文件路径')
    parser.add_argument('--test_file', type=str, default='./data/placement-0529-ga-20epoch-norotation/test.jsonl',
                        help='测试数据文件路径')
    parser.add_argument('--max_train_instances', type=int, default=50000,
                        help='最大训练实例数')
    parser.add_argument('--max_test_instances', type=int, default=2000,
                        help='最大测试实例数')
    
    # 模型相关参数
    parser.add_argument('--input_dim', type=int, default=2,
                        help='输入特征维度')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='隐藏层维度')
    parser.add_argument('--max_seq_len', type=int, default=60,
                        help='最大序列长度')
    
    # 训练相关参数
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='梯度裁剪最大范数')
    parser.add_argument('--use_baseline', action='store_true', default=True,
                        help='是否使用baseline减少方差')
    
    # 奖励权重
    parser.add_argument('--reward_weights', nargs=3, type=float, default=[0.4, 0.3, 0.3],
                        help='奖励权重 [紧凑度, 贴合度, 利用率]')
    
    # 输出相关参数
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存目录')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='模型保存间隔（轮数）')
    
    # wandb相关参数
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='是否使用wandb记录')
    parser.add_argument('--wandb_project', type=str, default='td_svgnest',
                        help='wandb项目名称')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='wandb运行名称')
    
    return parser.parse_args()


def main():
    """主训练函数"""
    args = parse_args()
    
    # 设置日志
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"训练参数: {args}")
    
    # 初始化wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # 检查数据文件
    if not os.path.exists(args.train_file):
        logger.error(f"训练数据文件不存在: {args.train_file}")
        return
    
    # 创建数据加载器
    train_loader = TDDataLoader(args.train_file)
    logger.info(f"加载了 {len(train_loader)} 个训练实例")
    
    # 创建测试数据加载器
    test_loader = None
    if os.path.exists(args.test_file):
        test_loader = TDDataLoader(args.test_file)
        logger.info(f"加载了 {len(test_loader)} 个测试实例")
    else:
        logger.warning(f"测试数据文件不存在: {args.test_file}")
    
    # 创建训练器
    trainer = TDTrainer(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        reward_weights=tuple(args.reward_weights),
        use_baseline=args.use_baseline,
        clip_grad_norm=args.clip_grad_norm,
        use_wandb=args.use_wandb
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练循环
    best_test_util = 0.0
    for epoch in range(args.num_epochs):
        logger.info(f"开始第 {epoch + 1}/{args.num_epochs} 轮训练...")
        
        # 训练一个epoch
        train_stats = trainer.train_epoch(train_loader, epoch, max_instances=args.max_train_instances)
        
        logger.info(f"第 {epoch + 1} 轮训练完成:")
        logger.info(f"  平均损失: {train_stats['avg_loss']:.4f}")
        logger.info(f"  平均奖励: {train_stats['avg_reward']:.4f}")
        logger.info(f"  平均利用率: {train_stats['avg_utilization']:.3f}")
        logger.info(f"  训练时间: {train_stats['epoch_time']:.2f}s")
        
        # 评估测试集
        if test_loader is not None:
            test_stats = trainer.evaluate(test_loader, num_instances=args.max_test_instances)
            logger.info(f"测试集评估结果:")
            logger.info(f"  平均奖励: {test_stats['avg_reward']:.4f}")
            logger.info(f"  平均利用率: {test_stats['avg_utilization']:.3f}")
            logger.info(f"  最佳利用率: {test_stats['best_utilization']:.3f}")
            logger.info(f"  评估时间: {test_stats['eval_time']:.2f}s")
            
            # 记录到wandb
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_stats['avg_loss'],
                    'train/reward': train_stats['avg_reward'],
                    'train/utilization': train_stats['avg_utilization'],
                    'train/baseline': train_stats['avg_baseline'],
                    'train/epoch_time': train_stats['epoch_time'],
                    'test/reward': test_stats['avg_reward'],
                    'test/utilization': test_stats['avg_utilization'],
                    'test/best_utilization': test_stats['best_utilization'],
                    'test/eval_time': test_stats['eval_time'],
                    'best/train_utilization': trainer.train_stats['best_utilization'],
                    'best/train_reward': trainer.train_stats['best_reward']
                })
            
            # 保存最佳模型
            is_best = test_stats['avg_utilization'] > best_test_util
            if is_best:
                best_test_util = test_stats['avg_utilization']
                
        else:
            # 记录到wandb（仅训练数据）
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_stats['avg_loss'],
                    'train/reward': train_stats['avg_reward'],
                    'train/utilization': train_stats['avg_utilization'],
                    'train/baseline': train_stats['avg_baseline'],
                    'train/epoch_time': train_stats['epoch_time'],
                    'best/train_utilization': trainer.train_stats['best_utilization'],
                    'best/train_reward': trainer.train_stats['best_reward']
                })
            
            is_best = train_stats['avg_utilization'] >= trainer.train_stats['best_utilization']
        
        # 保存模型
        if (epoch + 1) % args.save_interval == 0 or epoch == args.num_epochs - 1:
            model_path = os.path.join(args.output_dir, f'td_policy_epoch_{epoch + 1}.pth')
            trainer.save_model(model_path, epoch=epoch + 1, is_best=is_best)
    
    logger.info("训练完成!")
    
    # 关闭wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main()