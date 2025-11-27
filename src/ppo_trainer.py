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
from typing import Dict, Optional, List

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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

class EpisodeBuffer:
    """
    改进后的缓冲区，存储完整的 Episode 轨迹。
    """
    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.episodes = [] # 存储每个 episode 的字典

    def add_episode(self, episode_data: Dict):
        """
        episode_data 包含:
        - state: (max_seq_len, input_dim) 问题实例
        - actions: (seq_len,)
        - log_probs: (seq_len,)
        - rewards: (seq_len,)
        - values: (seq_len,)
        - valid_length: int
        """
        self.episodes.append(episode_data)

    def compute_gae_and_stack(self, gamma: float, gae_lambda: float):
        """
        为缓冲区中的所有 episodes 计算 GAE，并将它们堆叠成 Tensor Batch。
        """
        all_states = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []
        all_values = []
        all_valid_lengths = []
        
        if not self.episodes:
            return None

        # 获取 state 的形状以确定 max_seq_len
        # 注意：TDEnvironment 返回的 state 已经是固定大小 [max_parts, 2]
        max_len = self.episodes[0]['state'].shape[0]
        
        for ep in self.episodes:
            # 转换为 Tensor
            rewards = torch.tensor(ep['rewards'], dtype=torch.float32, device=self.device)
            values = torch.tensor(ep['values'], dtype=torch.float32, device=self.device)
            
            seq_len = len(rewards)
            advantages = torch.zeros_like(rewards)
            last_gae_lam = 0
            
            # GAE 计算
            for t in reversed(range(seq_len)):
                if t == seq_len - 1:
                    next_value = 0.0  # 假设 episode 结束时价值为 0
                else:
                    next_value = values[t + 1]
                    
                delta = rewards[t] + gamma * next_value - values[t]
                last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
                advantages[t] = last_gae_lam
            
            returns = advantages + values
            
            all_states.append(torch.tensor(ep['state'], dtype=torch.float32, device=self.device))
            all_actions.append(torch.tensor(ep['actions'], dtype=torch.long, device=self.device))
            all_log_probs.append(torch.tensor(ep['log_probs'], dtype=torch.float32, device=self.device))
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_values.append(values)
            all_valid_lengths.append(ep['valid_length'])

        # 堆叠并 Padding
        batch_size = len(self.episodes)
        
        padded_actions = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        padded_log_probs = torch.zeros((batch_size, max_len), dtype=torch.float32, device=self.device)
        padded_advantages = torch.zeros((batch_size, max_len), dtype=torch.float32, device=self.device)
        padded_returns = torch.zeros((batch_size, max_len), dtype=torch.float32, device=self.device)
        padded_values = torch.zeros((batch_size, max_len), dtype=torch.float32, device=self.device)

        for i, length in enumerate(all_valid_lengths):
            # 截断以防万一长度超过 max_len
            actual_len = min(length, max_len)
            padded_actions[i, :actual_len] = all_actions[i][:actual_len]
            padded_log_probs[i, :actual_len] = all_log_probs[i][:actual_len]
            padded_advantages[i, :actual_len] = all_advantages[i][:actual_len]
            padded_returns[i, :actual_len] = all_returns[i][:actual_len]
            padded_values[i, :actual_len] = all_values[i][:actual_len]

        return (
            torch.stack(all_states),      # [Batch, Max_Seq, Dim]
            padded_actions,               # [Batch, Max_Seq]
            padded_log_probs,             # [Batch, Max_Seq]
            padded_advantages,            # [Batch, Max_Seq]
            padded_returns,               # [Batch, Max_Seq]
            padded_values,                # [Batch, Max_Seq]
            torch.tensor(all_valid_lengths, device=self.device) # [Batch]
            )

    def __len__(self):
        return len(self.episodes)

class PPOTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 环境和数据加载器
        self.train_loader = TDDataLoader(args.train_file)
        self.test_loader = TDDataLoader(args.train_file, args.test_file) if args.test_file and os.path.exists(args.test_file) else None
        # 传入 rule_weight 以控制规则奖励强度
        self.env = TDEnvironment(
            max_parts=args.max_seq_len,
            reward_weights=tuple(args.reward_weights),
            rule_weight=args.rule_weight
        )
        
        # 模型
        self.policy = PPOActorCritic(args.input_dim, args.hidden_dim, args.max_seq_len).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.learning_rate, eps=1e-5)
        
        # 缓冲区 - 使用新的 EpisodeBuffer
        self.buffer = EpisodeBuffer(self.device)
        
        # 训练状态
        self.global_step = 0
        self.episode_count = 0
        self.best_eval_util = 0.0
        self.use_wandb = args.use_wandb and WANDB_AVAILABLE
        os.makedirs(args.output_dir, exist_ok=True)

    def learn(self):
        """主训练循环"""
        logger.info("Starting PPO training loop (Trajectory-based)...")

        while self.global_step < self.args.total_timesteps:
            # 1. 数据收集阶段 (Collection Phase)
            self.policy.eval()
            self.buffer.reset()

            collection_steps = 0
            episodes_collected = 0
            
            # 收集直到达到 batch_size 个 episodes
            target_episodes = self.args.batch_size 
            
            while episodes_collected < target_episodes:
                instance = self.train_loader.get_random_instance()
                state_np = self.env.reset(instance)
                current_state = torch.tensor(state_np, dtype=torch.float32, device=self.device)
                
                valid_length = self.env.num_parts
                valid_length_tensor = torch.tensor([valid_length], device=self.device)
                
                # 临时存储当前 episode 数据
                ep_actions = []
                ep_log_probs = []
                ep_rewards = []
                ep_values = []
                
                # 初始化 LSTM 和 Mask
                with torch.no_grad():
                    encoder_outputs = self.policy.encode_inputs(current_state.unsqueeze(0)).squeeze(0)
                decoder_input, decoder_hidden = self.policy.get_initial_state(1, self.device)
                decoder_input = decoder_input.squeeze(0)
                selected_mask = torch.zeros(self.args.max_seq_len, dtype=torch.bool, device=self.device)

                done = False
                ep_step = 0
                
                while not done:
                    self.global_step += 1
                    ep_step += 1
                    
                    with torch.no_grad():
                        # Forward 即使在 rollout 时也需要 mask 以防止重复选择
                        logits, value, next_decoder_hidden = self.policy.forward(
                            encoder_outputs.unsqueeze(0), decoder_input.unsqueeze(0),
                            decoder_hidden, selected_mask.unsqueeze(0), valid_length_tensor
                        )
                        
                        dist = Categorical(logits=logits)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                    
                    # 环境交互
                    next_state_np, reward, done, info = self.env.step(action.item())
                    
                    # 存储 Step 数据
                    ep_actions.append(action.item())
                    ep_log_probs.append(log_prob.item())
                    ep_rewards.append(reward)
                    ep_values.append(value.item())
                    
                    # 更新状态
                    decoder_hidden = next_decoder_hidden
                    decoder_input = encoder_outputs[action.item()]
                    selected_mask[action.item()] = True
                
                # Episode 结束，存入 Buffer
                self.buffer.add_episode({
                    'state': state_np, # 原始状态 (max_parts, 2)
                    'actions': ep_actions,
                    'log_probs': ep_log_probs,
                    'rewards': ep_rewards,
                    'values': ep_values,
                    'valid_length': valid_length
                })
                
                episodes_collected += 1
                collection_steps += ep_step
                self.episode_count += 1

                # 记录日志 (采样频率降低以避免刷屏)
                if self.episode_count % 10 == 0:
                    metrics = self.env.get_placement_metrics()
                    util = metrics.get('material_utilization', 0)
                    logger.info(f"Ep {self.episode_count}: Reward={sum(ep_rewards):.2f}, Len={ep_step}, Util={util:.3f}")
                    if self.use_wandb:
                        wandb.log({"rollout/ep_reward": sum(ep_rewards), "rollout/ep_util": util, "global_step": self.global_step})
                    
            # 2. 更新阶段 (Update Phase)
            logger.debug("Starting PPO update...")
            self.policy.train()
            
            # 计算 GAE 并获取堆叠后的 Batch 数据
            batch_data = self.buffer.compute_gae_and_stack(self.args.gamma, self.args.gae_lambda)
            if batch_data is None:
                continue
                
            states_b, actions_b, old_log_probs_b, advantages_b, returns_b, values_b, valid_lens_b = batch_data
            
            # 标准化优势
            if advantages_b.std() > 1e-8:
                advantages_b = (advantages_b - advantages_b.mean()) / (advantages_b.std() + 1e-8)
                    
            # PPO Epochs
            # 这里对整个 Batch 进行多轮更新
            for epoch in range(self.args.n_epochs):
                # 调用新的 evaluate_actions (处理整个序列，自动 Teacher Forcing 重建状态)
                new_log_probs, new_values, entropy = self.policy.evaluate_actions(states_b, actions_b, valid_lens_b)
                
                # 创建 Mask 以过滤掉 Padding 部分的影响
                # [Batch, Max_Seq]
                mask = torch.arange(self.args.max_seq_len, device=self.device).unsqueeze(0) < valid_lens_b.unsqueeze(1)
                mask_float = mask.float()
                
                # --- 计算 Loss ---
                # 1. Policy Loss
                ratio = torch.exp(new_log_probs - old_log_probs_b)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - self.args.clip_range, 1.0 + self.args.clip_range) * advantages_b
                policy_loss = -torch.min(surr1, surr2)
                
                # 应用 Mask 并取平均 (只对有效步数)
                policy_loss = (policy_loss * mask_float).sum() / (mask_float.sum() + 1e-8)

                # 2. Value Loss
                value_loss = nn.functional.mse_loss(new_values, returns_b, reduction='none')
                value_loss = (value_loss * mask_float).sum() / (mask_float.sum() + 1e-8)

                # 3. Entropy Loss
                entropy_loss = -entropy
                entropy_loss = (entropy_loss * mask_float).sum() / (mask_float.sum() + 1e-8)
                
                loss = policy_loss + self.args.vf_coef * value_loss + self.args.ent_coef * entropy_loss
                    
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
            
                if epoch == 0:
                    logger.debug(f"Update Epoch 0: Loss={loss.item():.4f}, P_Loss={policy_loss.item():.4f}, V_Loss={value_loss.item():.4f}")
            
            if self.use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/policy_loss": policy_loss.item(),
                    "train/value_loss": value_loss.item(),
                    "train/entropy_loss": entropy_loss.item(),
                    "global_step": self.global_step
                })

            # 定期评估
            if self.global_step % self.args.eval_interval < collection_steps and self.test_loader:
                self.evaluate()

    def evaluate(self):
        logger.info(f"Evaluating at step {self.global_step}...")
        self.policy.eval()
        total_util = 0.0
        num_instances = min(self.args.max_test_instances, len(self.test_loader.test_instances))

        for i in range(num_instances):
            instance = self.test_loader.get_test_instance(i)
            state_np = self.env.reset(instance)
            
            with torch.no_grad():
                state = torch.tensor(state_np, dtype=torch.float32, device=self.device)
                encoder_outputs = self.policy.encode_inputs(state.unsqueeze(0)).squeeze(0)
                decoder_input, decoder_hidden = self.policy.get_initial_state(1, self.device)
                decoder_input = decoder_input.squeeze(0)
                selected_mask = torch.zeros(self.args.max_seq_len, dtype=torch.bool, device=self.device)
                valid_length = torch.tensor([self.env.num_parts], device=self.device)

                done = False
                while not done:
                    logits, _, next_decoder_hidden = self.policy.forward(
                        encoder_outputs.unsqueeze(0), decoder_input.unsqueeze(0),
                        decoder_hidden, selected_mask.unsqueeze(0), valid_length
                    )
                    # 评估时使用贪心策略
                    action = torch.argmax(logits, dim=1)
                    
                    _, _, done, _ = self.env.step(action.item())
                    
                    decoder_hidden = next_decoder_hidden
                    decoder_input = encoder_outputs[action.item()]
                    selected_mask[action.item()] = True
            
            metrics = self.env.get_placement_metrics()
            total_util += metrics.get('material_utilization', 0)
        
        avg_util = total_util / num_instances
        logger.info(f"Eval Avg Util: {avg_util:.4f}")
        if self.use_wandb:
            wandb.log({"eval/avg_util": avg_util, "global_step": self.global_step})

        if avg_util > self.best_eval_util:
            self.best_eval_util = avg_util
            self.save_model(is_best=True)
        self.save_model(is_best=False)

    def save_model(self, is_best=False):
        filename = 'ppo_policy_best.pth' if is_best else f'ppo_policy_latest.pth'
        filepath = os.path.join(self.args.output_dir, filename)
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': self.args
        }, filepath)

def parse_args():
    parser = argparse.ArgumentParser(description='PPO Trainer for 2D Nesting Problem')
    
    # --- Data and Paths ---
    parser.add_argument('--train_file', type=str, default='./data/placement-0529-ga-20epoch-norotation/train.jsonl')
    parser.add_argument('--test_file', type=str, default='./data/placement-0529-ga-20epoch-norotation/test.jsonl')
    parser.add_argument('--output_dir', type=str, default='./models_ppo', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs_ppo', help='Directory to save logs')
    
    # --- Model Hyperparameters ---
    # 输入特征维度：宽、高、是否已放置、当前利用率、步数比例
    parser.add_argument('--input_dim', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--max_seq_len', type=int, default=60)
    
    # --- PPO Hyperparameters ---
    parser.add_argument('--total_timesteps', type=int, default=1_000_000, help='Total timesteps to train the agent')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    # 这里 batch_size 修改为 episodes per update
    parser.add_argument('--batch_size', type=int, default=16, help='Number of episodes per PPO update')
    parser.add_argument('--n_epochs', type=int, default=4, help='Number of epochs to update the policy on the collected data')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    
    # --- Environment and Reward ---
    parser.add_argument('--reward_weights', nargs=3, type=float, default=[0.65, 0.05, 0.3])
    parser.add_argument(
        '--rule_weight',
        type=float,
        default=0.05,
        help='规则奖励权重：鼓励接近面积降序的排样顺序，0 表示关闭规则奖励'
    )
    
    # --- Training and Logging ---
    parser.add_argument('--eval_interval', type=int, default=5000, help='Evaluate the agent every N steps')
    parser.add_argument('--max_test_instances', type=int, default=100)
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='ppo_svgnest')
    parser.add_argument('--wandb_run_name', type=str, default=f'ppo_run_{time.strftime("%Y%m%d-%H%M%S")}')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Setup Logging ---
    os.makedirs(args.log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(args.log_dir, 'train_ppo.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler]
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
    trainer.learn()
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

if __name__ == '__main__':
    main()
