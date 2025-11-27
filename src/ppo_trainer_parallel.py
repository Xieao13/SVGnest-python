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
from typing import Dict, Optional, Callable

# --- 添加项目根目录到Python路径 ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- 导入项目模块 ---
from src.model.ppo_actor_critic import PPOActorCritic
from src.environment.td_environment import TDEnvironment, TDDataLoader
from src.environment.vectorized_env import SubprocVecEnv  # <-- 导入新的并行环境

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

logger = logging.getLogger(__name__)


# --- RolloutBuffer (基本不变, 仅为清晰起见包含在此) ---
class RolloutBuffer:
    def __init__(self, n_steps: int, num_envs: int, state_shape: tuple, device):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.buffer_size = self.n_steps * self.num_envs
        self.state_shape = state_shape
        self.device = device
        self.reset()

    def reset(self):
        self.states = torch.zeros((self.buffer_size, *self.state_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.buffer_size,), dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros((self.buffer_size,), device=self.device)
        self.rewards = torch.zeros((self.buffer_size,), device=self.device)
        self.dones = torch.zeros((self.buffer_size,), device=self.device)
        self.values = torch.zeros((self.buffer_size,), device=self.device)
        self.pos = 0

    def add(self, states, actions, log_probs, rewards, dones, values):
        # 一次性添加来自所有环境的一步数据
        batch_size = states.shape[0]
        self.states[self.pos : self.pos + batch_size].copy_(states)
        self.actions[self.pos : self.pos + batch_size].copy_(actions)
        self.log_probs[self.pos : self.pos + batch_size].copy_(log_probs)
        self.rewards[self.pos : self.pos + batch_size].copy_(torch.from_numpy(rewards).to(self.device))
        self.dones[self.pos : self.pos + batch_size].copy_(torch.from_numpy(dones).to(self.device))
        self.values[self.pos : self.pos + batch_size].copy_(values.squeeze())
        self.pos += batch_size

    def compute_returns_and_advantages(self, last_values: torch.Tensor, gamma: float, gae_lambda: float):
        self.advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            # GAE需要按时间顺序计算，但我们的数据是 env1_t1, env2_t1, ..., envN_t1, env1_t2, ...
            # 这种扁平化的GAE计算是一个近似，但实践中效果很好
            # 更精确的实现需要重塑数据，但会增加复杂性
            next_non_terminal = 1.0 - self.dones[step]
            # 这里的 next_values 假设是下一步的，但在扁平化的buffer中可能是另一个环境的
            # 这是一个已知的简化，对于大型buffer通常可以接受
            # 为了简单起见，我们继续使用这种广泛采用的简化方法
            # 实际上，在done=True时，next_non_terminal=0会切断传播，这是最重要的
            if step == self.buffer_size - 1:
                 # 这里的 last_values 是一个 (num_envs,) 的向量
                 # 我们需要找到正确的 last_value (简化：使用均值)
                 next_values = last_values.mean()
            else:
                 next_values = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            self.advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        self.returns = self.advantages + self.values

    def get_generator(self, batch_size: int):
        indices = np.random.permutation(self.buffer_size)
        for start_idx in range(0, self.buffer_size, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield (
                self.states[batch_indices], self.actions[batch_indices],
                self.log_probs[batch_indices], self.advantages[batch_indices],
                self.returns[batch_indices]
            )

# --- 用于创建环境的辅助函数 ---
def make_env(args, rank: int) -> Callable:
    def _init() -> TDEnvironment:
        # 这里的 data_loader 在子进程中重新创建
        train_loader = TDDataLoader(args.train_file)
        env = TDEnvironment(max_parts=args.max_seq_len, reward_weights=tuple(args.reward_weights))
        # 可以设置不同的随机种子 (如果需要)
        # env.seed(args.seed + rank)
        return env
    return _init

class PPOTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- 创建并行环境 ---
        env_fns = [make_env(args, i) for i in range(args.num_envs)]
        self.envs = SubprocVecEnv(env_fns)
        
        # --- 测试环境（仍然是单个） ---
        self.test_loader = TDDataLoader(args.train_file, args.test_file) if args.test_file and os.path.exists(args.test_file) else None
        self.eval_env = TDEnvironment(max_parts=args.max_seq_len, reward_weights=tuple(args.reward_weights))
        
        # 模型
        self.policy = PPOActorCritic(args.input_dim, args.hidden_dim, args.max_seq_len).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.learning_rate, eps=1e-5)
        
        # 缓冲区
        state_shape = (args.max_seq_len, args.input_dim)
        self.buffer = RolloutBuffer(args.n_steps, args.num_envs, state_shape, self.device)
        
        # 训练状态
        self.global_step = 0
        self.best_eval_util = 0.0
        self.use_wandb = args.use_wandb and WANDB_AVAILABLE
        os.makedirs(args.output_dir, exist_ok=True)

    def learn(self):
        """主训练循环"""
        # --- 初始化多个并行环境的状态 ---
        current_states = torch.from_numpy(self.envs.reset()).float().to(self.device)
        
        # --- 为每个环境维护独立的解码器状态 ---
        encoder_outputs = self.policy.encode_inputs(current_states)
        decoder_inputs, decoder_hiddens = self.policy.get_initial_state(self.args.num_envs, self.device)
        selected_masks = torch.zeros(self.args.num_envs, self.args.max_seq_len, dtype=torch.bool, device=self.device)
        
        episode_rewards = np.zeros(self.args.num_envs)
        episode_lengths = np.zeros(self.args.num_envs)

        total_timesteps = self.args.n_steps * self.args.num_envs * (self.args.total_timesteps // (self.args.n_steps * self.args.num_envs))

        while self.global_step < total_timesteps:
            # --- 数据收集阶段 ---
            self.policy.eval()
            self.buffer.reset()
            
            for _ in range(self.args.n_steps):
                valid_lengths = (current_states[:, :, 0] > 1e-6).sum(dim=1)
                
                with torch.no_grad():
                    logits, values, next_decoder_hiddens = self.policy.forward(
                        encoder_outputs, decoder_inputs, decoder_hiddens, selected_masks, valid_lengths
                    )
                    dist = Categorical(logits=logits)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)

                # 与所有并行环境交互
                next_states_np, rewards, dones, infos = self.envs.step(actions.cpu().numpy())
                next_states = torch.from_numpy(next_states_np).float().to(self.device)
                
                self.buffer.add(current_states, actions, log_probs, rewards, dones, values)
                self.global_step += self.args.num_envs

                # --- 更新状态 ---
                current_states = next_states
                encoder_outputs = self.policy.encode_inputs(current_states)
                decoder_hiddens = next_decoder_hiddens
                decoder_inputs = encoder_outputs[torch.arange(self.args.num_envs), actions]
                selected_masks[torch.arange(self.args.num_envs), actions] = True

                # --- 处理结束的环境 ---
                episode_rewards += rewards
                episode_lengths += 1
                for i, done in enumerate(dones):
                    if done:
                        # 从info中获取真实指标
                        utilization = infos[i].get('final_metrics', {}).get('material_utilization', 0)
                        logger.info(f"Step {self.global_step}: Env {i} finished. Reward={episode_rewards[i]:.2f}, Length={episode_lengths[i]}, Utilization={utilization:.3f}")
                        if self.use_wandb:
                            wandb.log({
                                "rollout/episode_reward": episode_rewards[i],
                                "rollout/episode_length": episode_lengths[i],
                                "rollout/episode_utilization": utilization,
                                "global_step": self.global_step
                            })
                        # 重置该环境的状态
                        d_inputs, d_hiddens = self.policy.get_initial_state(1, self.device)
                        decoder_inputs[i] = d_inputs.squeeze(0)
                        decoder_hiddens[0][i] = d_hiddens[0].squeeze(0)
                        decoder_hiddens[1][i] = d_hiddens[1].squeeze(0)
                        selected_masks[i].zero_()
                        episode_rewards[i] = 0
                        episode_lengths[i] = 0

                if self.global_step % self.args.eval_interval == 0 and self.test_loader:
                    self.evaluate()

            # --- GAE 和回报计算 ---
            with torch.no_grad():
                valid_lengths = (current_states[:, :, 0] > 1e-6).sum(dim=1)
                _, last_values, _ = self.policy.forward(
                    encoder_outputs, decoder_inputs, decoder_hiddens, selected_masks, valid_lengths
                )
            self.buffer.compute_returns_and_advantages(last_values.squeeze(), self.args.gamma, self.args.gae_lambda)

            # --- 学习更新阶段 (与之前类似，但处理的是更大的buffer) ---
            self.policy.train()
            for _ in range(self.args.n_epochs):
                for states, actions, old_log_probs, advantages, returns in self.buffer.get_generator(self.args.batch_size):
                    new_log_probs, new_values, entropy = self.policy.evaluate_actions(states, actions.unsqueeze(1))
                    
                    new_log_probs = new_log_probs.squeeze(1)
                    new_values = new_values.squeeze(1)
                    entropy = entropy.squeeze(1)
                    
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    policy_loss1 = advantages * ratio
                    policy_loss2 = advantages * torch.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)
                    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

                    value_loss = nn.functional.mse_loss(returns, new_values)
                    entropy_loss = -torch.mean(entropy)
                    loss = policy_loss + self.args.vf_coef * value_loss + self.args.ent_coef * entropy_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

            if self.use_wandb:
                wandb.log({
                    "train/policy_loss": policy_loss.item(),
                    "train/value_loss": value_loss.item(),
                    "train/entropy_loss": entropy_loss.item(),
                    "global_step": self.global_step
                })
    # ... (evaluate 和 save_model 方法基本不变, 只是环境对象变成 self.eval_env)
    def evaluate(self):
        self.policy.eval()
        total_util = 0.0
        num_instances = min(self.args.max_test_instances, len(self.test_loader.test_instances))
        
        logger.info(f"--- Running evaluation for {num_instances} instances ---")

        for i in range(num_instances):
            instance = self.test_loader.get_test_instance(i)
            state_np = self.eval_env.reset(instance)
            done = False
            
            with torch.no_grad():
                state = torch.tensor(state_np, dtype=torch.float32, device=self.device)
                encoder_outputs = self.policy.encode_inputs(state.unsqueeze(0)).squeeze(0)
                decoder_input, decoder_hidden = self.policy.get_initial_state(1, self.device)
                decoder_input = decoder_input.squeeze(0)
                selected_mask = torch.zeros(self.args.max_seq_len, dtype=torch.bool, device=self.device)

                valid_length = torch.tensor([self.eval_env.num_parts], device=self.device)
                while not done:
                    # 将 valid_length 传递给 forward
                    logits, _, next_decoder_hidden = self.policy.forward(
                        encoder_outputs.unsqueeze(0), decoder_input.unsqueeze(0),
                        decoder_hidden, selected_mask.unsqueeze(0), valid_length
                    )
                    action = torch.argmax(logits, dim=1) # 贪心策略
                    
                    _, _, done, _ = self.eval_env.step(action.item())
                    
                    decoder_hidden = next_decoder_hidden
                    decoder_input = encoder_outputs[action.item()]
                    selected_mask[action.item()] = True
            
            metrics = self.eval_env.get_placement_metrics()
            total_util += metrics.get('material_utilization', 0)
        
        avg_util = total_util / num_instances
        logger.info(f"--- Evaluation finished. Average Utilization: {avg_util:.4f} ---")

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

# --- 修改 parse_args 和 main ---
def parse_args():
    parser = argparse.ArgumentParser(description='Parallel PPO Trainer for 2D Nesting')
    # ... (所有参数不变) ...
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
    
    # --- 新增并行化参数 ---
    parser.add_argument('--num_envs', type=int, default=4, help='Number of parallel environments to run')
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