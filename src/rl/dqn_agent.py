import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import List, Tuple, Optional
import math


class DQNNetwork(nn.Module):
    """DQN网络架构"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super(DQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 网络架构
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 双流架构：值函数和优势函数
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 特征提取
        features = self.feature_extractor(state)
        
        # 计算值函数和优势函数
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.Experience = namedtuple(
            'Experience', 
            ['state', 'action', 'reward', 'next_state', 'done']
        )
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        experience = self.Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """采样批次"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN智能体"""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000,
        target_update_freq: int = 1000,
        buffer_size: int = 100000,
        batch_size: int = 32,
        hidden_dim: int = 512,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = device
        
        # 网络
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # 初始化目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), 
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 训练统计
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_losses = []
    
    def get_epsilon(self) -> float:
        """获取当前epsilon值"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               math.exp(-1. * self.steps_done / self.epsilon_decay)
    
    def select_action(self, state: np.ndarray, valid_actions: List[int] = None) -> int:
        """选择动作"""
        if valid_actions is None:
            valid_actions = list(range(self.action_dim))
            
        # epsilon-贪婪策略
        epsilon = self.get_epsilon()
        
        if random.random() > epsilon:
            # 贪婪动作选择
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                
                # 只考虑有效动作
                masked_q_values = q_values.clone()
                invalid_actions = [a for a in range(self.action_dim) if a not in valid_actions]
                if invalid_actions:
                    masked_q_values[0, invalid_actions] = float('-inf')
                
                action = masked_q_values.argmax().item()
        else:
            # 随机动作选择
            action = random.choice(valid_actions)
        
        self.steps_done += 1
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转换"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 奖励归一化和裁剪
        # 裁剪奖励到合理范围 (匹配新的奖励scale)
        rewards = torch.clamp(rewards, min=-10.0, max=200.0)  # 匹配新的奖励范围0-170
        
        # 当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 下一状态的Q值（使用目标网络）
        with torch.no_grad():
            # Double DQN: 使用策略网络选择动作，目标网络评估
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones).unsqueeze(1))
            
            # 目标值裁剪，防止Q值膨胀 (匹配新的scale)
            target_q_values = torch.clamp(target_q_values, min=-50.0, max=400.0)  # 而不是-10到50
        
        # 计算损失
        loss = F.huber_loss(current_q_values, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # 更新目标网络
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_losses = checkpoint.get('episode_losses', [])
    
    def set_eval_mode(self):
        """设置为评估模式"""
        self.policy_net.eval()
    
    def set_train_mode(self):
        """设置为训练模式"""
        self.policy_net.train()
    
    def get_stats(self) -> dict:
        """获取训练统计信息"""
        stats = {
            'steps_done': self.steps_done,
            'epsilon': self.get_epsilon(),
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_loss_last_100': np.mean(self.episode_losses[-100:]) if self.episode_losses else 0,
            'total_episodes': len(self.episode_rewards)
        }
        
        return stats 