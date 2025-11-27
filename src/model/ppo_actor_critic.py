import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Optional

class PPOActorCritic(nn.Module):
    """
    适用于PPO的Actor-Critic网络，集成了Pointer Network。
    - Actor: Pointer Network的解码器，用于选择下一个零件。
    - Critic: 一个简单的MLP，用于评估当前状态的价值。
    """
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(PPOActorCritic, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # --- 共享编码器 ---
        self.part_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.pos_encoder = nn.Embedding(seq_len, hidden_dim // 4)
        
        # 确保拼接后维度正确
        combined_dim = (hidden_dim // 2) + (hidden_dim // 4)
        self.encoder_fc = nn.Linear(combined_dim, hidden_dim) # 添加一个FC层来统一维度

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 2,
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # --- Actor (Pointer Network 解码器) ---
        self.decoder_rnn = nn.LSTMCell(hidden_dim, hidden_dim)
        self.pointer_head = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pointer_v = nn.Linear(hidden_dim, 1, bias=False)

        # --- Critic (价值头) ---
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def encode_inputs(self, parts: torch.Tensor) -> torch.Tensor:
        """
        对输入的零件序列进行编码，生成用于Actor和Critic的上下文向量。
        """
        batch_size, seq_len_actual, _ = parts.shape
        
        parts_encoded = self.part_encoder(parts)
        positions = torch.arange(seq_len_actual, device=parts.device).unsqueeze(0).expand(batch_size, -1)
        pos_encoded = self.pos_encoder(positions)
        
        combined = torch.cat([parts_encoded, pos_encoded], dim=-1)
        # 使用FC层将拼接后的特征映射到hidden_dim
        combined_mapped = self.encoder_fc(combined)

        encoder_output = self.transformer_encoder(combined_mapped)
        return encoder_output

    def forward(self, encoder_outputs: torch.Tensor, decoder_input: torch.Tensor,
                decoder_hidden: tuple, selected_mask: torch.Tensor, valid_lengths: Optional[torch.Tensor] = None):
        """
        执行一步前向传播。
        """
        # --- Actor ---
        # 更新LSTM隐藏状态
        h, c = self.decoder_rnn(decoder_input, decoder_hidden)
        decoder_output = h.unsqueeze(1) # [batch_size, 1, hidden_dim]

        # 计算Pointer Network的logits
        # 使用广播机制: [batch_size, 1, hidden_dim] + [batch_size, seq_len, hidden_dim]
        combined_features = torch.tanh(
            self.pointer_head(decoder_output) + self.pointer_head(encoder_outputs)
        )
        logits = self.pointer_v(combined_features).squeeze(-1) # [batch_size, seq_len]
        
        # 应用已选择零件的mask (防止重复选择)
        if selected_mask is not None:
            logits.masked_fill_(selected_mask, -1e8)

        # --- 应用 padding mask (防止选择不存在的零件) ---
        if valid_lengths is not None:
            # 创建一个范围张量 [0, 1, 2, ..., seq_len-1]
            range_tensor = torch.arange(encoder_outputs.size(1), device=encoder_outputs.device).unsqueeze(0)
            # 广播 valid_lengths 来创建mask
            padding_mask = range_tensor >= valid_lengths.unsqueeze(1)
            logits.masked_fill_(padding_mask, -1e8)
        # --------------------------------

        # --- Critic ---
        # 使用解码器的隐藏状态作为当前状态的表征来计算价值
        value = self.critic_head(h)

        return logits, value, (h, c)

    def get_initial_state(self, batch_size, device):
        """获取解码器的初始隐藏状态和输入"""
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        # 初始解码器输入为一个零向量，代表 "START"
        initial_decoder_input = torch.zeros(batch_size, self.hidden_dim, device=device)
        return initial_decoder_input, (h0, c0)

    def evaluate_actions(self, parts: torch.Tensor, actions: torch.Tensor, valid_lengths: torch.Tensor):
        """
        核心修复：基于完整序列的评估。
        """
        batch_size, max_seq_len, _ = parts.shape
        device = parts.device

        # 1. 编码整个问题实例
        encoder_outputs = self.encode_inputs(parts)
        
        # 2. 初始化解码器状态
        decoder_input, decoder_hidden = self.get_initial_state(batch_size, device)
        
        # 3. 初始化 Mask (全 False)
        selected_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)
        
        all_log_probs = []
        all_values = []
        all_entropies = []
        
        # 4. 逐步重演序列生成过程
        for t in range(max_seq_len):
            # 获取当前步的真实动作 [batch_size]
            current_actions = actions[:, t]
            
            # 关键修正：显式创建一个新的 Mask Tensor，而不是 inplace 修改
            # Step 1: 确保当前动作未被 mask
            # 使用 clone 创建副本，确保不影响梯度计算
            mask_for_step = selected_mask.clone()
            mask_for_step[torch.arange(batch_size), current_actions] = False
            
            # 前向传播
            logits, value, decoder_hidden = self.forward(
                encoder_outputs, decoder_input, decoder_hidden, mask_for_step, valid_lengths
            )
            
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(current_actions)
            entropy = dist.entropy()
            
            all_log_probs.append(log_prob)
            all_values.append(value.squeeze(1))
            all_entropies.append(entropy)
            
            # 准备下一步的输入
            # 1. 更新 Mask：生成下一个时间步的基础 Mask
            # 同样，为了避免 inplace operation 错误，我们基于 mask_for_step 创建新的 mask
            next_mask = mask_for_step.clone()
            next_mask[torch.arange(batch_size), current_actions] = True
            selected_mask = next_mask # 更新引用
            
            # 2. 更新 Decoder Input：使用 Teacher Forcing (使用历史真实动作)
            decoder_input = encoder_outputs[torch.arange(batch_size), current_actions]

        # 堆叠结果 [batch_size, max_seq_len]
        return (torch.stack(all_log_probs, dim=1), 
                torch.stack(all_values, dim=1), 
                torch.stack(all_entropies, dim=1))
