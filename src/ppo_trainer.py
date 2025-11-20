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
        self.rewards = torch.zeros((self.n_steps,), device=self.device)
        self.dones = torch.zeros((self.n_steps,), device=self.device)
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
        
        # 模型
        self.policy = PPOActorCritic(args.input_dim, args.hidden_dim, args.max_seq_len).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.learning_rate, eps=1e-5)
        
        # 缓冲区
        state_shape = (args.max_seq_len, args.input_dim)
        self.buffer = RolloutBuffer(args.n_steps, state_shape, self.device)
        
        # 训练状态
        self.global_step = 0
        self.best_eval_util = 0.0
        self.use_wandb = args.use_wandb and WANDB_AVAILABLE
        os.makedirs(args.output_dir, exist_ok=True)

    def learn(self):
        """主训练循环"""
        start_time = time.time()
        
        # --- 初始化环境状态 ---
        instance = self.train_loader.get_random_instance()
        logger.debug(f"=== Initial instance loaded: bin={instance['bin']}, num_parts={len(instance['parts'])}")
        
        current_state = torch.tensor(self.env.reset(instance), dtype=torch.float32, device=self.device)
        logger.debug(f"Initial state shape: {current_state.shape}, env.num_parts={self.env.num_parts}")
        
        # 注意：这里的encoder_outputs, decoder_input等状态需要在每次环境重置后都更新
        
        episode_reward = 0
        episode_len = 0

        while self.global_step < self.args.total_timesteps:
            # --- 数据收集阶段 ---
            logger.debug(f"\n{'='*60}\nStarting rollout at step {self.global_step}")
            self.policy.eval()
            self.buffer.reset()

            # 对整个问题实例进行一次编码，这个编码在当前episode中共享
            with torch.no_grad():
                encoder_outputs = self.policy.encode_inputs(current_state.unsqueeze(0)).squeeze(0)
            logger.debug(f"Encoder outputs shape: {encoder_outputs.shape}")
            
            # 初始化解码器状态
            decoder_input, decoder_hidden = self.policy.get_initial_state(1, self.device)
            decoder_input = decoder_input.squeeze(0)
            selected_mask = torch.zeros(self.args.max_seq_len, dtype=torch.bool, device=self.device)

            # 获取当前 episode 的有效长度
            valid_length = torch.tensor([self.env.num_parts], device=self.device)
            logger.debug(f"Valid length for current instance: {valid_length.item()}")

            for step_idx in range(self.args.n_steps):
                self.global_step += 1
                
                with torch.no_grad():
                    # 将 valid_length 传递给 forward
                    logits, value, next_decoder_hidden = self.policy.forward(
                        encoder_outputs.unsqueeze(0), decoder_input.unsqueeze(0),
                        decoder_hidden, selected_mask.unsqueeze(0), valid_length
                    )
                    
                    # Debug: 输出logits统计信息
                    if step_idx < 5 or self.global_step % 500 == 0:  # 只在前5步或每500步输出
                        logger.debug(f"Step {self.global_step}: logits range=[{logits.min().item():.3f}, {logits.max().item():.3f}], "
                                   f"mean={logits.mean().item():.3f}, value={value.item():.3f}")
                        # 输出前5个最高概率的动作
                        probs = torch.softmax(logits, dim=-1).squeeze()
                        top_probs, top_indices = torch.topk(probs, min(5, valid_length.item()))
                        logger.debug(f"  Top 5 actions: indices={top_indices.tolist()}, probs={top_probs.tolist()}")
                    
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    if step_idx < 5 or self.global_step % 500 == 0:
                        logger.debug(f"  Selected action: {action.item()}, log_prob: {log_prob.item():.3f}")
                        # Debug: 检查mask状态
                        num_selected = selected_mask.sum().item()
                        logger.debug(f"  Mask status: {num_selected} parts already selected, mask={selected_mask.nonzero(as_tuple=True)[0].tolist()[:10]}")  # 只显示前10个

                # 与环境交互
                next_state_np, reward, done, info = self.env.step(action.item())
                
                # Debug: 输出奖励信息
                if step_idx < 5 or self.global_step % 500 == 0:
                    logger.debug(f"  Reward: {reward:.4f}, Done: {done}, Invalid: {info.get('invalid_action', False)}")
                    if info.get('invalid_action', False):
                        logger.debug(f"  ⚠️ INVALID ACTION DETECTED! Action {action.item()} was already selected.")
                        logger.debug(f"  Current placement_sequence: {info.get('placement_sequence', [])}")
                
                # 添加到缓冲区
                self.buffer.add(current_state, action, log_prob, reward, done, value)
                
                # 为下一步准备状态 (在检查done之前)
                current_state = torch.tensor(next_state_np, dtype=torch.float32, device=self.device)
                decoder_hidden = next_decoder_hidden
                decoder_input = encoder_outputs[action.item()]
                selected_mask[action.item()] = True
                
                # Debug: 验证mask更新
                if step_idx < 5 or self.global_step % 500 == 0:
                    if not selected_mask[action.item()]:
                        logger.debug(f"  ⚠️ WARNING: Mask was not updated correctly for action {action.item()}!")
                
                episode_reward += reward
                episode_len += 1

                # --- 这是核心修改逻辑 ---
                if done:
                    # 1. 记录完成的Episode信息
                    metrics = self.env.get_placement_metrics()
                    utilization = metrics.get('material_utilization', 0)
                    logger.info(f"Step {self.global_step}: Episode finished. Reward={episode_reward:.2f}, Length={episode_len}, Utilization={utilization:.3f}")
                    logger.debug(f"  Episode details - Placement sequence: {info.get('placement_sequence', [])}")
                    logger.debug(f"  Full metrics: {metrics}")
                    if self.use_wandb:
                        wandb.log({
                            "rollout/episode_reward": episode_reward,
                            "rollout/episode_length": episode_len,
                            "rollout/episode_utilization": utilization,
                            "global_step": self.global_step
                        })
                    
                    # 2. 重置环境和智能体状态，为下一个Episode做准备
                    # 重置环境和智能体状态
                    instance = self.train_loader.get_random_instance()
                    logger.debug(f"  New instance loaded: bin={instance['bin']}, num_parts={len(instance['parts'])}")
                    current_state = torch.tensor(self.env.reset(instance), dtype=torch.float32, device=self.device)
                    
                    # --- 新增：重置后立即更新 valid_length ---
                    valid_length = torch.tensor([self.env.num_parts], device=self.device)
                    logger.debug(f"  Reset: valid_length={valid_length.item()}, state_shape={current_state.shape}")
                    # ------------------------------------------
                    
                    # 重新编码新问题的状态
                    with torch.no_grad():
                        encoder_outputs = self.policy.encode_inputs(current_state.unsqueeze(0)).squeeze(0)
                    
                    
                    # 重置解码器状态
                    decoder_input, decoder_hidden = self.policy.get_initial_state(1, self.device)
                    decoder_input = decoder_input.squeeze(0)
                    selected_mask.zero_()
                    
                    # Debug: 验证重置后的状态
                    logger.debug(f"  After reset: env.placement_sequence={self.env.placement_sequence}, "
                               f"mask_sum={selected_mask.sum().item()}, "
                               f"env.num_parts={self.env.num_parts}")

                    # 3. 重置Episode的追踪器
                    episode_reward, episode_len = 0, 0
                
                # 定期评估 (这个逻辑不变)
                if self.global_step % self.args.eval_interval == 0 and self.test_loader:
                    self.evaluate()

            # --- GAE 和回报计算 (这个逻辑不变) ---
            with torch.no_grad():
                # 计算最后一个状态的价值用于bootstrapping
                # 注意：如果一个episode正好在n_steps的最后一步结束，这里的last_value应该是0，但为了简化，我们依然计算
                # 这是一个微小的近似，在实践中影响不大
                _, last_value, _ = self.policy.forward(
                    encoder_outputs.unsqueeze(0), decoder_input.unsqueeze(0),
                    decoder_hidden, selected_mask.unsqueeze(0)
                )
            logger.debug(f"Computing GAE with last_value={last_value.item():.3f}")
            self.buffer.compute_returns_and_advantages(last_value, self.args.gamma, self.args.gae_lambda)
            
            # Debug: 输出advantages和returns的统计信息
            logger.debug(f"GAE computed - advantages: mean={self.buffer.advantages.mean().item():.3f}, "
                       f"std={self.buffer.advantages.std().item():.3f}, "
                       f"min={self.buffer.advantages.min().item():.3f}, "
                       f"max={self.buffer.advantages.max().item():.3f}")
            logger.debug(f"Returns: mean={self.buffer.returns.mean().item():.3f}, "
                       f"std={self.buffer.returns.std().item():.3f}")

            # --- 学习更新阶段 (!!! 这里是核心修改 !!!) ---
            logger.debug(f"\nStarting policy update: {self.args.n_epochs} epochs, batch_size={self.args.batch_size}")
            self.policy.train()
            
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_total_losses = []
            epoch_ratios = []
            
            for epoch_idx in range(self.args.n_epochs):
                batch_idx = 0
                # 使用 get_generator 来获取 mini-batch 数据
                for states, actions, old_log_probs, advantages, returns in self.buffer.get_generator(self.args.batch_size):
                    batch_idx += 1
                    # states shape: (batch_size, 60, 2)
                    # actions shape: (batch_size,)
                    
                    # 我们将 actions 扩展一个维度，使其符合 evaluate_actions 的输入要求
                    # actions.unsqueeze(1) shape: (batch_size, 1)
                    # 这意味着我们将每个 (state, action) 对看作是一个长度为1的序列
                    new_log_probs, new_values, entropy = self.policy.evaluate_actions(states, actions.unsqueeze(1))
                    
                    # new_log_probs, new_values, entropy 的 shape 都是 (batch_size, 1)
                    # 我们需要 .squeeze() 来匹配 buffer 中存储的 1D 张量
                    new_log_probs = new_log_probs.squeeze(1)
                    new_values = new_values.squeeze(1)
                    entropy = entropy.squeeze(1)
                    
                    # Debug: 比较rollout和更新阶段的log_prob（第一个epoch的前3个batch）
                    if epoch_idx == 0 and batch_idx <= 3:
                        log_prob_diff = (new_log_probs - old_log_probs).abs()
                        logger.debug(f"  Log prob comparison: old_mean={old_log_probs.mean().item():.3f}, "
                                   f"new_mean={new_log_probs.mean().item():.3f}, "
                                   f"diff_mean={log_prob_diff.mean().item():.3f}, "
                                   f"max_diff={log_prob_diff.max().item():.3f}")
                    
                    # 标准化优势 (可选但推荐)
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # --- 计算 PPO 损失 ---
                    # 策略损失
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    policy_loss1 = advantages * ratio
                    policy_loss2 = advantages * torch.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)
                    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

                    # 价值损失 (可以裁剪，也可以不裁剪，这里使用简单的MSE)
                    value_loss = nn.functional.mse_loss(returns, new_values)

                    # 熵损失
                    entropy_loss = -torch.mean(entropy)

                    # 总损失
                    loss = policy_loss + self.args.vf_coef * value_loss + self.args.ent_coef * entropy_loss
                    
                    # 记录损失和ratio统计
                    epoch_policy_losses.append(policy_loss.item())
                    epoch_value_losses.append(value_loss.item())
                    epoch_total_losses.append(loss.item())
                    epoch_ratios.extend(ratio.detach().cpu().tolist())
                    
                    # Debug: 输出第一个epoch的前3个batch的详细信息
                    if epoch_idx == 0 and batch_idx <= 3:
                        logger.debug(f"  Epoch {epoch_idx}, Batch {batch_idx}:")
                        logger.debug(f"    Ratio: mean={ratio.mean().item():.3f}, std={ratio.std().item():.3f}, "
                                   f"min={ratio.min().item():.3f}, max={ratio.max().item():.3f}")
                        logger.debug(f"    Losses: policy={policy_loss.item():.4f}, value={value_loss.item():.4f}, "
                                   f"entropy={entropy_loss.item():.4f}, total={loss.item():.4f}")
                        clipped_ratio = torch.sum((ratio < 1 - self.args.clip_range) | (ratio > 1 + self.args.clip_range)).item()
                        logger.debug(f"    Clipped ratios: {clipped_ratio}/{ratio.numel()} ({100*clipped_ratio/ratio.numel():.1f}%)")
                    
                    # --- 优化 ---
                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                    
                    if epoch_idx == 0 and batch_idx <= 3:
                        logger.debug(f"    Gradient norm before clipping: {grad_norm:.3f}")
                    
                    self.optimizer.step()
            
            # Debug: 输出整个更新周期的统计信息
            avg_policy_loss = np.mean(epoch_policy_losses)
            avg_value_loss = np.mean(epoch_value_losses)
            avg_total_loss = np.mean(epoch_total_losses)
            avg_ratio = np.mean(epoch_ratios)
            std_ratio = np.std(epoch_ratios)
            logger.debug(f"\nUpdate summary - Average losses: policy={avg_policy_loss:.4f}, value={avg_value_loss:.4f}, total={avg_total_loss:.4f}")
            logger.debug(f"Ratio statistics: mean={avg_ratio:.3f}, std={std_ratio:.3f}")
            logger.info(f"Step {self.global_step}: Update completed. Avg total loss={avg_total_loss:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    "train/policy_loss": avg_policy_loss,
                    "train/value_loss": avg_value_loss,
                    "train/entropy_loss": -np.mean([e for e in epoch_total_losses]),  # approximation
                    "train/total_loss": avg_total_loss,
                    "train/ratio_mean": avg_ratio,
                    "train/ratio_std": std_ratio,
                    "global_step": self.global_step
                })
    def evaluate(self):
        logger.debug(f"\n{'='*60}\nStarting evaluation at step {self.global_step}")
        self.policy.eval()
        total_util = 0.0
        num_instances = min(self.args.max_test_instances, len(self.test_loader.test_instances))
        
        logger.info(f"--- Running evaluation for {num_instances} instances ---")
        
        # 记录前几个实例的详细信息
        detailed_results = []

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
                    # 将 valid_length 传递给 forward
                    logits, _, next_decoder_hidden = self.policy.forward(
                        encoder_outputs.unsqueeze(0), decoder_input.unsqueeze(0),
                        decoder_hidden, selected_mask.unsqueeze(0), valid_length
                    )
                    action = torch.argmax(logits, dim=1) # 贪心策略
                    
                    _, _, done, _ = self.env.step(action.item())
                    
                    decoder_hidden = next_decoder_hidden
                    decoder_input = encoder_outputs[action.item()]
                    selected_mask[action.item()] = True
            
            metrics = self.env.get_placement_metrics()
            util = metrics.get('material_utilization', 0)
            total_util += util
            
            # 记录前5个实例的详细结果
            if i < 5:
                detailed_results.append({
                    'instance_id': i,
                    'utilization': util,
                    'bbox_ratio': metrics.get('bbox_ratio', 0),
                    'waste_area': metrics.get('waste_area', 0)
                })
                logger.debug(f"  Instance {i}: util={util:.4f}, bbox_ratio={metrics.get('bbox_ratio', 0):.4f}")
        
        avg_util = total_util / num_instances
        logger.info(f"--- Evaluation finished. Average Utilization: {avg_util:.4f} ---")
        logger.debug(f"Evaluation details (first 5): {detailed_results}")

        if self.use_wandb:
            wandb.log({
                "eval/avg_utilization": avg_util,
                "global_step": self.global_step
            })

        if avg_util > self.best_eval_util:
            self.best_eval_util = avg_util
            logger.info(f"New best evaluation utilization: {self.best_eval_util:.4f}. Saving model.")
            self.save_model(is_best=True)
        
        self.save_model(is_best=False) # 保存最新模型

    def save_model(self, is_best=False):
        filename = f'ppo_policy_step_{self.global_step}.pth'
        if is_best:
            filename = 'ppo_policy_best.pth'
        
        filepath = os.path.join(self.args.output_dir, filename)
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
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
    
    # --- PPO Hyperparameters ---
    parser.add_argument('--total_timesteps', type=int, default=1_000_000, help='Total timesteps to train the agent')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps to run for each environment per update')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size for PPO updates')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to update the policy on the collected data')
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
    
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Setup Logging ---
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建两个handler：一个用于DEBUG级别的文件日志，一个用于INFO级别的控制台输出
    file_handler = logging.FileHandler(os.path.join(args.log_dir, 'train_ppo.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # 控制台只显示INFO及以上
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 配置根logger
    logging.basicConfig(
        level=logging.DEBUG,  # 设置为DEBUG以捕获所有日志
        handlers=[file_handler, console_handler]
    )
    
    logger.info(f"Training arguments: {args}")
    logger.debug("Debug logging is enabled. Detailed logs will be written to file.")
    
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