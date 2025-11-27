import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os
import time
import sys
import argparse
from torch.distributions import Categorical
from typing import Dict, Optional
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- 导入项目模块 ---
from src.model.ppo_actor_critic import PPOActorCritic
from src.environment.td_environment import TDEnvironment, TDDataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

logger = logging.getLogger(__name__)

class RolloutBuffer:
    """用于存储PPO rollout数据的缓冲区"""
    def __init__(self, n_steps: int, state_shape: tuple, device):
        self.n_steps = n_steps
        self.state_shape = state_shape
        self.device = device
        self.reset()

    def reset(self):
        self.states = torch.zeros((self.n_steps, *self.state_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.n_steps,), dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros((self.n_steps,), device=self.device)
        self.rewards = torch.zeros((self.n_steps,), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.n_steps,), dtype=torch.bool, device=self.device)
        self.values = torch.zeros((self.n_steps,), device=self.device)
        self.advantages = torch.zeros((self.n_steps,), device=self.device)
        self.returns = torch.zeros((self.n_steps,), device=self.device)
        self.pos = 0
        self.is_full = False

    def add(self, state, action, log_prob, reward, done, value):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = torch.tensor(reward, device=self.device, dtype=torch.float32)
        self.dones[self.pos] = torch.tensor(done, device=self.device, dtype=torch.float32)
        self.values[self.pos] = value.squeeze()
        self.pos += 1
        if self.pos == self.n_steps:
            self.is_full = True

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - self.dones[step]
            next_values = self.values[step + 1] if step < self.n_steps - 1 else last_value
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def get_generator(self, batch_size: int):
        indices = np.random.permutation(self.n_steps)
        for start_idx in range(0, self.n_steps, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield (
                self.states[batch_indices], self.actions[batch_indices],
                self.log_probs[batch_indices], self.advantages[batch_indices],
                self.returns[batch_indices]
            )

class PPOTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 环境和数据加载器
        self.train_loader = TDDataLoader(args.train_file)
        self.test_loader = TDDataLoader(args.train_file, args.test_file) if args.test_file and os.path.exists(args.test_file) else None
        self.env = TDEnvironment(max_parts=args.max_seq_len, reward_weights=tuple(args.reward_weights))
        
        # 计算训练相关数量
        self.num_train_instances = len(self.train_loader.instances)
        self.total_instances = self.num_train_instances * args.num_epochs
        self.instances_per_rollout = args.n_steps // args.avg_episode_length  # 估算每个rollout处理多少个instance
        if self.instances_per_rollout == 0:
            self.instances_per_rollout = 1
        self.total_rollouts = max(1, self.total_instances // self.instances_per_rollout)
        
        logger.info(f"Training setup: {self.num_train_instances} instances x {args.num_epochs} epochs = {self.total_instances} total instances")
        logger.info(f"Estimated {self.total_rollouts} rollouts needed")
        
        # 模型
        self.policy = PPOActorCritic(args.input_dim, args.hidden_dim, args.max_seq_len).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.learning_rate, eps=1e-5)
        
        # 缓冲区
        state_shape = (args.max_seq_len, args.input_dim)
        self.buffer = RolloutBuffer(args.n_steps, state_shape, self.device)
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.instances_trained = 0
        self.rollout_count = 0
        self.best_eval_util = 0.0
        self.use_wandb = args.use_wandb and WANDB_AVAILABLE
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Episode统计相关
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_utilizations = []
        
        # 日志打印控制
        self.log_episode_interval = args.log_episode_interval
        self.last_log_time = time.time()
        self.log_time_interval = args.log_time_interval

    def learn(self):
        """主训练循环 - 基于epoch和数据量"""
        start_time = time.time()
        
        # 创建进度条 - 基于rollout数量而不是时间步数
        progress_bar = tqdm(total=self.total_rollouts, desc="PPO Training", unit="rollouts")
        
        # 训练实例索引管理
        instance_indices = list(range(self.num_train_instances))
        current_instance_idx = 0
        
        # 初始化第一个instance
        instance = self.train_loader.instances[instance_indices[current_instance_idx]]
        current_state = torch.tensor(self.env.reset(instance), dtype=torch.float32, device=self.device)
        
        episode_reward = 0
        episode_len = 0

        while self.current_epoch < self.args.num_epochs:
            # --- 数据收集阶段 ---
            self.policy.eval()
            self.buffer.reset()
            rollout_instances_count = 0

            # 对整个问题实例进行一次编码
            with torch.no_grad():
                encoder_outputs = self.policy.encode_inputs(current_state.unsqueeze(0)).squeeze(0)
            
            # 初始化解码器状态
            decoder_input, decoder_hidden = self.policy.get_initial_state(1, self.device)
            decoder_input = decoder_input.squeeze(0)
            selected_mask = torch.zeros(self.args.max_seq_len, dtype=torch.bool, device=self.device)
            valid_length = torch.tensor([self.env.num_parts], device=self.device)

            for step in range(self.args.n_steps):
                self.global_step += 1
                
                with torch.no_grad():
                    logits, value, next_decoder_hidden = self.policy.forward(
                        encoder_outputs.unsqueeze(0), decoder_input.unsqueeze(0),
                        decoder_hidden, selected_mask.unsqueeze(0), valid_length
                    )
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                # 与环境交互
                next_state_np, reward, done, _ = self.env.step(action.item())
                
                # 添加到缓冲区
                self.buffer.add(current_state, action, log_prob, reward, done, value)
                
                # 更新状态
                current_state = torch.tensor(next_state_np, dtype=torch.float32, device=self.device)
                decoder_hidden = next_decoder_hidden
                decoder_input = encoder_outputs[action.item()]
                selected_mask[action.item()] = True
                
                episode_reward += reward
                episode_len += 1

                # Episode完成处理
                if done:
                    self.episode_count += 1
                    rollout_instances_count += 1
                    self.instances_trained += 1
                    
                    metrics = self.env.get_placement_metrics()
                    utilization = metrics.get('material_utilization', 0)
                    
                    # 记录Episode统计信息
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_len)
                    self.episode_utilizations.append(utilization)
                    
                    # 控制详细日志打印频率
                    if self.episode_count % self.log_episode_interval == 0:
                        self._log_episode_details(episode_reward, episode_len, utilization)
                    
                    # W&B日志记录
                    if self.use_wandb:
                        wandb.log({
                            "rollout/episode_reward": episode_reward,
                            "rollout/episode_length": episode_len,
                            "rollout/episode_utilization": utilization,
                            "rollout/episode_count": self.episode_count,
                            "rollout/instances_trained": self.instances_trained,
                            "rollout/current_epoch": self.current_epoch,
                            "global_step": self.global_step
                        })
                    
                    # 获取下一个训练实例
                    current_instance_idx += 1
                    if current_instance_idx >= self.num_train_instances:
                        # 完成一个epoch，重新shuffle数据
                        current_instance_idx = 0
                        self.current_epoch += 1
                        np.random.shuffle(instance_indices)  # shuffle训练数据
                        logger.info(f"Completed epoch {self.current_epoch-1}, starting epoch {self.current_epoch}")
                        
                        # 检查是否完成所有epoch
                        if self.current_epoch >= self.args.num_epochs:
                            logger.info("All epochs completed!")
                            break
                    
                    # 重置环境到下一个实例
                    instance = self.train_loader.instances[instance_indices[current_instance_idx]]
                    current_state = torch.tensor(self.env.reset(instance), dtype=torch.float32, device=self.device)
                    
                    # 重新编码新实例
                    with torch.no_grad():
                        encoder_outputs = self.policy.encode_inputs(current_state.unsqueeze(0)).squeeze(0)
                    
                    # 重置解码器状态
                    decoder_input, decoder_hidden = self.policy.get_initial_state(1, self.device)
                    decoder_input = decoder_input.squeeze(0)
                    selected_mask.zero_()
                    valid_length = torch.tensor([self.env.num_parts], device=self.device)

                    # 重置Episode追踪器
                    episode_reward, episode_len = 0, 0
                
                # 定期评估
                if self.global_step % self.args.eval_interval == 0 and self.test_loader:
                    self.evaluate()
                
                # 检查是否完成所有epoch
                if self.current_epoch >= self.args.num_epochs:
                    break

            # 如果完成了所有epoch，退出主循环
            if self.current_epoch >= self.args.num_epochs:
                break

            # 更新rollout计数和进度条
            self.rollout_count += 1
            progress_bar.update(1)
            
            # 更新进度条显示信息
            if len(self.episode_rewards) >= 10:
                recent_avg_reward = np.mean(self.episode_rewards[-10:])
                recent_avg_util = np.mean(self.episode_utilizations[-10:])
                progress_pct = (self.instances_trained / self.total_instances) * 100
                progress_bar.set_postfix({
                    'Epoch': f'{self.current_epoch}/{self.args.num_epochs}',
                    'Progress': f'{progress_pct:.1f}%',
                    'Episodes': self.episode_count,
                    'Avg_R': f'{recent_avg_reward:.2f}',
                    'Avg_U': f'{recent_avg_util:.3f}',
                    'Instances': f'{rollout_instances_count}'
                })
            
            # 按时间间隔打印概要信息
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_time_interval:
                self._log_training_summary(current_time - start_time)
                self.last_log_time = current_time

            # --- GAE 和回报计算 ---
            with torch.no_grad():
                _, last_value, _ = self.policy.forward(
                    encoder_outputs.unsqueeze(0), decoder_input.unsqueeze(0),
                    decoder_hidden, selected_mask.unsqueeze(0), valid_length
                )
            self.buffer.compute_returns_and_advantages(last_value, self.args.gamma, self.args.gae_lambda)

            # --- 学习更新阶段 ---
            self.policy.train()
            for _ in range(self.args.ppo_epochs):
                for states, actions, old_log_probs, advantages, returns in self.buffer.get_generator(self.args.batch_size):
                    new_log_probs, new_values, entropy = self.policy.evaluate_actions(states, actions.unsqueeze(1))
                    
                    new_log_probs = new_log_probs.squeeze(1)
                    new_values = new_values.squeeze(1)
                    entropy = entropy.squeeze(1)
                    
                    # 标准化优势
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # 计算 PPO 损失
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    policy_loss1 = advantages * ratio
                    policy_loss2 = advantages * torch.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)
                    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

                    value_loss = nn.functional.mse_loss(returns, new_values)
                    entropy_loss = -torch.mean(entropy)
                    loss = policy_loss + self.args.vf_coef * value_loss + self.args.ent_coef * entropy_loss
                    
                    # 优化
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
            
            if self.use_wandb:
                wandb.log({
                    "train/policy_loss": policy_loss.item(),
                    "train/value_loss": value_loss.item(),
                    "train/entropy_loss": entropy_loss.item(),
                    "train/total_loss": loss.item(),
                    "train/rollout_count": self.rollout_count,
                    "global_step": self.global_step
                })
        
        progress_bar.close()
        logger.info(f"Training completed! Processed {self.instances_trained} instances over {self.current_epoch} epochs")
        
    def _log_episode_details(self, episode_reward, episode_len, utilization):
        """打印详细的episode信息"""
        recent_rewards = self.episode_rewards[-self.log_episode_interval:]
        recent_lengths = self.episode_lengths[-self.log_episode_interval:]
        recent_utils = self.episode_utilizations[-self.log_episode_interval:]
        
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        avg_util = np.mean(recent_utils)
        
        logger.info(
            f"Episode {self.episode_count} (Epoch {self.current_epoch}): "
            f"Last {self.log_episode_interval} episodes - "
            f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}, "
            f"Avg Util: {avg_util:.3f} | Current: R={episode_reward:.2f}, "
            f"L={episode_len}, U={utilization:.3f}"
        )
        
    def _log_training_summary(self, elapsed_time):
        """打印训练概要信息"""
        if len(self.episode_rewards) == 0:
            return
            
        recent_count = min(100, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-recent_count:]
        recent_utils = self.episode_utilizations[-recent_count:]
        
        progress_pct = (self.instances_trained / self.total_instances) * 100
        instances_per_sec = self.instances_trained / elapsed_time if elapsed_time > 0 else 0
        eta_seconds = (self.total_instances - self.instances_trained) / instances_per_sec if instances_per_sec > 0 else 0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        
        logger.info(
            f"Training Progress: {progress_pct:.1f}% | "
            f"Epoch: {self.current_epoch}/{self.args.num_epochs} | "
            f"Instances: {self.instances_trained}/{self.total_instances} | "
            f"Episodes: {self.episode_count} | "
            f"Speed: {instances_per_sec:.1f} instances/s | "
            f"ETA: {eta_str} | "
            f"Recent {recent_count} episodes - "
            f"Avg Reward: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}, "
            f"Avg Util: {np.mean(recent_utils):.3f} ± {np.std(recent_utils):.3f}"
        )

    def evaluate(self):
        self.policy.eval()
        total_util = 0.0
        num_instances = min(self.args.max_test_instances, len(self.test_loader.test_instances))
        
        logger.info(f"--- Running evaluation for {num_instances} instances ---")

        for i in range(num_instances):
            instance = self.test_loader.get_test_instance(i)
            state_np = self.env.reset(instance)
            done = False
            
            with torch.no_grad():
                state = torch.tensor(state_np, dtype=torch.float32, device=self.device)
                encoder_outputs = self.policy.encode_inputs(state.unsqueeze(0)).squeeze(0)
                decoder_input, decoder_hidden = self.policy.get_initial_state(1, self.device)
                decoder_input = decoder_input.squeeze(0)
                selected_mask = torch.zeros(self.args.max_seq_len, dtype=torch.bool, device=self.device)

                valid_length = torch.tensor([self.env.num_parts], device=self.device)
                while not done:
                    logits, _, next_decoder_hidden = self.policy.forward(
                        encoder_outputs.unsqueeze(0), decoder_input.unsqueeze(0),
                        decoder_hidden, selected_mask.unsqueeze(0), valid_length
                    )
                    action = torch.argmax(logits, dim=1)
                    
                    _, _, done, _ = self.env.step(action.item())
                    
                    decoder_hidden = next_decoder_hidden
                    decoder_input = encoder_outputs[action.item()]
                    selected_mask[action.item()] = True
            
            metrics = self.env.get_placement_metrics()
            total_util += metrics.get('material_utilization', 0)
        
        avg_util = total_util / num_instances
        logger.info(f"--- Evaluation finished. Average Utilization: {avg_util:.4f} ---")

        if self.use_wandb:
            wandb.log({
                "eval/avg_utilization": avg_util,
                "eval/current_epoch": self.current_epoch,
                "global_step": self.global_step
            })

        if avg_util > self.best_eval_util:
            self.best_eval_util = avg_util
            logger.info(f"New best evaluation utilization: {self.best_eval_util:.4f}. Saving model.")
            self.save_model(is_best=True)
        
        self.save_model(is_best=False)

    def save_model(self, is_best=False):
        filename = f'ppo_policy_epoch_{self.current_epoch}_step_{self.global_step}.pth'
        if is_best:
            filename = 'ppo_policy_best.pth'
        
        filepath = os.path.join(self.args.output_dir, filename)
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'instances_trained': self.instances_trained,
            'args': self.args
        }, filepath)
        logger.info(f"Model saved to: {filepath}")

def parse_args():
    parser = argparse.ArgumentParser(description='PPO Trainer for 2D Nesting Problem')
    
    # --- Data and Paths ---
    parser.add_argument('--train_file', type=str, default='./data/placement-0529-ga-20epoch-norotation/train.jsonl')
    parser.add_argument('--test_file', type=str, default='./data/placement-0529-ga-20epoch-norotation/test.jsonl')
    parser.add_argument('--output_dir', type=str, default='./models_ppo', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs_ppo', help='Directory to save logs')
    
    # --- Model Hyperparameters ---
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--max_seq_len', type=int, default=60)
    
    # --- Training Data Control (替换 total_timesteps) ---
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train (will process all training data)')
    parser.add_argument('--avg_episode_length', type=int, default=50, help='Average episode length for rollout estimation')
    
    # --- PPO Hyperparameters ---
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps to run for each rollout')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size for PPO updates')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='Number of epochs to update the policy on the collected data')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.0, help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    
    # --- Environment and Reward ---
    parser.add_argument('--reward_weights', nargs=3, type=float, default=[0.65, 0.05, 0.3])
    
    # --- Training and Logging ---
    parser.add_argument('--eval_interval', type=int, default=10000, help='Evaluate the agent every N steps')
    parser.add_argument('--max_test_instances', type=int, default=1000)
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='ppo_svgnest')
    parser.add_argument('--wandb_run_name', type=str, default=f'ppo_run_{time.strftime("%Y%m%d-%H%M%S")}')
    
    # --- 日志控制参数 ---
    parser.add_argument('--log_episode_interval', type=int, default=50, help='Print detailed episode info every N episodes')
    parser.add_argument('--log_time_interval', type=int, default=300, help='Print training summary every N seconds')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Setup Logging ---
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'train_ppo.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger.info(f"Training arguments: {args}")
    
    # --- Setup W&B ---
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # --- Create and run trainer ---
    trainer = PPOTrainer(args)
    logger.info("Starting PPO training...")
    trainer.learn()
    logger.info("Training finished!")
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

if __name__ == '__main__':
    main()