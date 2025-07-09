import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


logger = logging.getLogger(__name__)


class TDPolicyNetwork(nn.Module):
    """基于时序差分的策略网络 - 移除bin编码，只使用零件特征"""
    
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(TDPolicyNetwork, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # 零件特征编码器
        self.part_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # 位置编码
        self.pos_encoder = nn.Embedding(seq_len, hidden_dim // 4)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Pointer Network解码器
        self.decoder_rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Attention机制用于pointer network
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 输出层
        self.pointer_head = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_v = nn.Linear(hidden_dim, 1, bias=False)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=0.1)
                else:
                    nn.init.uniform_(param, -0.01, 0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def encode_inputs(self, parts):
        """编码输入特征 - 只使用零件特征"""
        batch_size, seq_len_actual = parts.shape[:2]
        
        # 编码零件特征
        parts_encoded = self.part_encoder(parts)  # [batch_size, seq_len, hidden_dim//2]
        
        # 位置编码
        positions = torch.arange(seq_len_actual, device=parts.device).unsqueeze(0).expand(batch_size, -1)
        pos_encoded = self.pos_encoder(positions)  # [batch_size, seq_len, hidden_dim//4]
        
        # 拼接特征 - 需要调整维度匹配
        # parts_encoded: [batch_size, seq_len, hidden_dim//2]
        # pos_encoded: [batch_size, seq_len, hidden_dim//4]
        # 需要拼接到hidden_dim维度
        padding_size = self.hidden_dim - parts_encoded.size(-1) - pos_encoded.size(-1)
        if padding_size > 0:
            padding = torch.zeros(batch_size, seq_len_actual, padding_size, device=parts.device)
            combined = torch.cat([parts_encoded, pos_encoded, padding], dim=-1)
        else:
            combined = torch.cat([parts_encoded, pos_encoded], dim=-1)
        
        # Transformer编码
        encoder_output = self.transformer_encoder(combined)
        
        return encoder_output

    def forward(self, parts, target_sequence=None, valid_lengths=None, teacher_forcing_ratio=1.0):
        """前向传播 - 时序差分版本"""
        batch_size, seq_len_padded = parts.shape[:2]
        device = parts.device
        
        # 编码输入
        encoder_outputs = self.encode_inputs(parts)  # [batch_size, seq_len, hidden_dim]
        
        # 初始化解码器状态
        decoder_hidden = (
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
            torch.zeros(1, batch_size, self.hidden_dim, device=device)
        )
        
        # 存储所有时间步的输出
        all_logits = []
        decoder_input = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
        
        # 用于tracking已选择的零件
        if target_sequence is not None:
            # 训练时使用teacher forcing - 只循环到target_sequence的长度
            target_seq_len = target_sequence.size(1)
            for step in range(target_seq_len):
                # 解码器一步
                decoder_output, decoder_hidden = self.decoder_rnn(decoder_input, decoder_hidden)

                # Pointer network logits计算
                combined_features = torch.tanh(
                    self.pointer_head(decoder_output) + self.pointer_head(encoder_outputs)
                )  # [batch_size, seq_len_padded, hidden_dim]
                step_logits = self.pointer_v(combined_features).squeeze(-1)  # [batch_size, seq_len_padded]
                
                # 如果提供了valid_lengths，mask掉padding部分
                if valid_lengths is not None:
                    padding_mask = torch.zeros_like(step_logits, dtype=torch.bool)
                    for i, valid_len in enumerate(valid_lengths):
                        padding_mask[i, valid_len:] = True
                    step_logits.masked_fill_(padding_mask, float('-inf'))
                
                all_logits.append(step_logits)

                # Teacher forcing: 使用真实的下一个选择作为输入
                if step < target_seq_len - 1 and target_sequence is not None:
                    if torch.rand(1).item() < teacher_forcing_ratio:
                        next_idx = target_sequence[:, step]
                        decoder_input = encoder_outputs[torch.arange(batch_size), next_idx].unsqueeze(1)
                    else:
                        # 使用模型预测的结果
                        next_idx = torch.argmax(step_logits, dim=1)
                        decoder_input = encoder_outputs[torch.arange(batch_size), next_idx].unsqueeze(1)
                else:
                    next_idx = torch.argmax(step_logits, dim=1)
                    decoder_input = encoder_outputs[torch.arange(batch_size), next_idx].unsqueeze(1)
        else:
            # 推理时的贪心解码
            all_logits = self._greedy_decode(encoder_outputs, decoder_hidden, seq_len_padded, valid_lengths)
        
        # 堆叠所有时间步的logits
        return torch.stack(all_logits, dim=1)  # [batch_size, seq_len_padded, seq_len_padded]

    def _greedy_decode(self, encoder_outputs, decoder_hidden, seq_len, valid_lengths):
        """贪心解码"""
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        all_logits = []
        selected_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        decoder_input = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
        
        for step in range(seq_len):
            # 解码器一步
            decoder_output, decoder_hidden = self.decoder_rnn(decoder_input, decoder_hidden)
            
            # 计算所有零件的logits
            step_logits = []
            for i in range(seq_len):
                combined_features = torch.tanh(
                    self.pointer_head(decoder_output) + 
                    self.pointer_head(encoder_outputs[:, i:i+1, :])
                )
                part_logit = self.pointer_v(combined_features).squeeze(-1)
                step_logits.append(part_logit)
            
            step_logits = torch.cat(step_logits, dim=1)  # [batch_size, seq_len]
            
            # 应用mask - 已选择的零件设为-inf
            step_logits.masked_fill_(selected_mask, float('-inf'))
            
            # 如果提供了valid_lengths，mask掉padding部分
            if valid_lengths is not None:
                padding_mask = torch.zeros_like(step_logits, dtype=torch.bool)
                for i, valid_len in enumerate(valid_lengths):
                    padding_mask[i, valid_len:] = True
                step_logits.masked_fill_(padding_mask, float('-inf'))
            
            all_logits.append(step_logits)
            
            # 选择最佳零件
            if step < seq_len - 1:
                next_idx = torch.argmax(step_logits, dim=1)
                # 更新mask
                selected_mask[torch.arange(batch_size), next_idx] = True
                # 更新decoder input
                decoder_input = encoder_outputs[torch.arange(batch_size), next_idx].unsqueeze(1)
        
        return all_logits

    def predict(self, parts, valid_lengths=None):
        """推理模式的预测"""
        with torch.no_grad():
            batch_size, seq_len_padded = parts.shape[:2]
            device = parts.device
            
            # 编码输入
            encoder_outputs = self.encode_inputs(parts)
            
            # 初始化
            decoder_hidden = (
                torch.zeros(1, batch_size, self.hidden_dim, device=device),
                torch.zeros(1, batch_size, self.hidden_dim, device=device)
            )
            
            predictions = []
            selected_mask = torch.zeros(batch_size, seq_len_padded, device=device, dtype=torch.bool)
            decoder_input = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
            
            for step in range(seq_len_padded):
                # 解码器一步
                decoder_output, decoder_hidden = self.decoder_rnn(decoder_input, decoder_hidden)
                
                # 计算所有零件的logits
                step_logits = []
                for i in range(seq_len_padded):
                    combined_features = torch.tanh(
                        self.pointer_head(decoder_output) + 
                        self.pointer_head(encoder_outputs[:, i:i+1, :])
                    )
                    part_logit = self.pointer_v(combined_features).squeeze(-1)
                    step_logits.append(part_logit)
                
                step_logits = torch.cat(step_logits, dim=1)  # [batch_size, seq_len_padded]
                
                # 应用mask
                step_logits.masked_fill_(selected_mask, float('-inf'))
                
                # 如果提供了valid_lengths，mask掉padding部分
                if valid_lengths is not None:
                    padding_mask = torch.zeros_like(step_logits, dtype=torch.bool)
                    for i, valid_len in enumerate(valid_lengths):
                        padding_mask[i, valid_len:] = True
                    step_logits.masked_fill_(padding_mask, float('-inf'))
                
                # 选择最佳零件
                next_idx = torch.argmax(step_logits, dim=1)
                predictions.append(next_idx)
                
                # 更新状态
                if step < seq_len_padded - 1:
                    selected_mask[torch.arange(batch_size), next_idx] = True
                    decoder_input = encoder_outputs[torch.arange(batch_size), next_idx].unsqueeze(1)
            
            return torch.stack(predictions, dim=1)  # [batch_size, seq_len_padded]

    def sample_action(self, parts, valid_lengths=None, temperature=1.0):
        """采样动作 - 用于时序差分训练"""
        with torch.no_grad():
            batch_size, seq_len_padded = parts.shape[:2]
            device = parts.device
            
            # 编码输入
            encoder_outputs = self.encode_inputs(parts)
            
            # 初始化
            decoder_hidden = (
                torch.zeros(1, batch_size, self.hidden_dim, device=device),
                torch.zeros(1, batch_size, self.hidden_dim, device=device)
            )
            
            actions = []
            log_probs = []
            selected_mask = torch.zeros(batch_size, seq_len_padded, device=device, dtype=torch.bool)
            decoder_input = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
            
            for step in range(seq_len_padded):
                # 解码器一步
                decoder_output, decoder_hidden = self.decoder_rnn(decoder_input, decoder_hidden)
                
                # 计算所有零件的logits
                step_logits = []
                for i in range(seq_len_padded):
                    combined_features = torch.tanh(
                        self.pointer_head(decoder_output) + 
                        self.pointer_head(encoder_outputs[:, i:i+1, :])
                    )
                    part_logit = self.pointer_v(combined_features).squeeze(-1)
                    step_logits.append(part_logit)
                
                step_logits = torch.cat(step_logits, dim=1)  # [batch_size, seq_len_padded]
                
                # 应用mask
                step_logits.masked_fill_(selected_mask, float('-inf'))
                
                # 如果提供了valid_lengths，mask掉padding部分
                if valid_lengths is not None:
                    padding_mask = torch.zeros_like(step_logits, dtype=torch.bool)
                    for i, valid_len in enumerate(valid_lengths):
                        padding_mask[i, valid_len:] = True
                    step_logits.masked_fill_(padding_mask, float('-inf'))
                
                # 温度采样 - 确保数值稳定性
                step_logits_scaled = step_logits / temperature
                
                # 创建有效动作的mask
                valid_action_mask = ~(torch.isinf(step_logits_scaled) | torch.isnan(step_logits_scaled))
                
                # 如果所有动作都无效，随机选择一个
                if not torch.any(valid_action_mask):
                    # 创建uniform分布
                    probs = torch.ones_like(step_logits_scaled)
                    probs = probs / probs.sum(dim=1, keepdim=True)
                else:
                    # 只对有效动作计算softmax
                    masked_logits = step_logits_scaled.clone()
                    masked_logits[~valid_action_mask] = float('-inf')
                    probs = F.softmax(masked_logits, dim=1)
                
                # 确保概率有效且非负
                probs = torch.clamp(probs, min=1e-8, max=1.0)
                probs = probs / probs.sum(dim=1, keepdim=True)
                
                action = torch.multinomial(probs, 1).squeeze(1)
                
                # 计算log概率
                log_prob = F.log_softmax(step_logits_scaled, dim=1)
                action_log_prob = log_prob.gather(1, action.unsqueeze(1)).squeeze(1)
                
                actions.append(action)
                log_probs.append(action_log_prob)
                
                # 更新状态
                if step < seq_len_padded - 1:
                    selected_mask[torch.arange(batch_size), action] = True
                    decoder_input = encoder_outputs[torch.arange(batch_size), action].unsqueeze(1)
            
            return torch.stack(actions, dim=1), torch.stack(log_probs, dim=1)  # [batch_size, seq_len_padded] 