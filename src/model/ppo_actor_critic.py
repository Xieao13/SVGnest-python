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

        # --- 共享编码器 (来自你的 TDPolicyNetwork) ---
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
        执行一步前向传播，用于PPO的 rollout 过程。

        Args:
            encoder_outputs: 编码器的输出 [batch_size, seq_len, hidden_dim]
            decoder_input: 上一步选择的零件的编码 [batch_size, hidden_dim]
            decoder_hidden: LSTM的隐藏状态
            selected_mask: 已选择零件的mask [batch_size, seq_len]

        Returns:
            logits: 指向每个零件的分数 [batch_size, seq_len]
            value: 当前状态的价值 [batch_size, 1]
            decoder_hidden: 更新后的LSTM隐藏状态
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
        
        # 应用已选择零件的mask
        logits.masked_fill_(selected_mask, float('-inf'))

        # --- 新增：应用 padding mask ---
        if valid_lengths is not None:
            # 创建一个范围张量 [0, 1, 2, ..., seq_len-1]
            range_tensor = torch.arange(encoder_outputs.size(1), device=encoder_outputs.device).unsqueeze(0)
            # 广播 valid_lengths 来创建mask
            padding_mask = range_tensor >= valid_lengths.unsqueeze(1)
            logits.masked_fill_(padding_mask, float('-inf'))
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

    def evaluate_actions(self, parts: torch.Tensor, actions: torch.Tensor):
        """
        在PPO的更新阶段，重新评估一批(state, action)对的log_prob, value和entropy。
        
        注意：这个函数假设actions是一个序列，但实际上在PPO中，每个(state, action)对是独立的。
        当max_steps=1时，每个state只对应一个action，mask应该从空开始。
        但是，这会导致问题：在rollout时，mask可能已经包含了一些已选零件，但在更新时mask是空的。
        """
        batch_size = parts.size(0)
        max_steps = actions.size(1)
        device = parts.device

        # --- 新增：从 parts 张量计算 valid_lengths ---
        # 假设 padding 的部分特征值为0
        # (parts > 0) 会对每个特征检查，我们只需要检查一个即可，比如宽度
        valid_lengths = (parts[:, :, 0] > 1e-6).sum(dim=1)
        # ------------------------------------------

        encoder_outputs = self.encode_inputs(parts)
        decoder_input, decoder_hidden = self.get_initial_state(batch_size, device)
        
        # ⚠️ 问题：mask总是从空开始，但在rollout时，mask可能已经包含了一些已选零件
        selected_mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=device)
        
        log_probs, values, entropies = [], [], []
        
        for i in range(max_steps):
            action_step = actions[:, i]
            
            # Debug: 检查mask状态（只在第一个batch的前几步输出）
            if i == 0 and batch_size > 0:
                num_selected = selected_mask[0].sum().item()
                # 只在mask不为空时输出（说明有已选零件）
                if num_selected > 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"evaluate_actions: step {i}, mask has {num_selected} selected parts")
            
            # 传递 valid_lengths
            logits, value, decoder_hidden = self.forward(
                encoder_outputs, decoder_input, decoder_hidden, selected_mask, valid_lengths
            )
            
            # 计算分布和指标
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action_step)
            entropy = dist.entropy()
            
            log_probs.append(log_prob)
            values.append(value.squeeze(1))
            entropies.append(entropy)
            
            # 更新下一个时间步的输入和mask
            decoder_input = encoder_outputs[torch.arange(batch_size), action_step]
            # selected_mask[torch.arange(batch_size), action_step] = True
            next_mask = selected_mask.clone().detach()
            next_mask[torch.arange(batch_size), action_step] = True
            selected_mask = next_mask

        return torch.stack(log_probs, dim=1), torch.stack(values, dim=1), torch.stack(entropies, dim=1)