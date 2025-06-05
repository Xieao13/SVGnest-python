import numpy as np
import torch
import torch.nn as nn
import os
import json
import logging
import sys
import random

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class PlacementOnlyNetwork(nn.Module):
    """改进的放置顺序网络 - 实现真正的autoregressive预测"""
    
    def __init__(self, input_dim, hidden_dim, seq_len, bin_dim=2):
        super(PlacementOnlyNetwork, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.bin_dim = bin_dim
        
        # 零件特征编码器
        self.part_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # 容器特征编码器
        self.bin_encoder = nn.Sequential(
            nn.Linear(bin_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
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

    def encode_inputs(self, parts, bin_info):
        """编码输入特征"""
        batch_size, seq_len_actual = parts.shape[:2]
        
        # 编码零件特征
        parts_encoded = self.part_encoder(parts)  # [batch_size, seq_len, hidden_dim//2]
        
        # 编码容器特征
        bin_encoded = self.bin_encoder(bin_info)  # [batch_size, hidden_dim//4]
        bin_encoded = bin_encoded.unsqueeze(1).expand(-1, seq_len_actual, -1)
        
        # 位置编码
        positions = torch.arange(seq_len_actual, device=parts.device).unsqueeze(0).expand(batch_size, -1)
        pos_encoded = self.pos_encoder(positions)  # [batch_size, seq_len, hidden_dim//4]
        
        # 拼接特征
        combined = torch.cat([parts_encoded, bin_encoded, pos_encoded], dim=-1)
        
        # Transformer编码
        encoder_output = self.transformer_encoder(combined)
        
        return encoder_output

    def forward(self, parts, bin_info, target_sequence=None, teacher_forcing_ratio=1.0):
        """前向传播 - 真正的autoregressive预测"""
        batch_size, seq_len_actual = parts.shape[:2]
        device = parts.device
        
        # 编码输入
        encoder_outputs = self.encode_inputs(parts, bin_info)  # [batch_size, seq_len, hidden_dim]
        
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
            # 训练时使用teacher forcing
            for step in range(seq_len_actual):
                # 解码器一步
                decoder_output, decoder_hidden = self.decoder_rnn(decoder_input, decoder_hidden)
                
                # 计算attention权重 (pointer mechanism)
                query = decoder_output  # [batch_size, 1, hidden_dim]
                key = encoder_outputs  # [batch_size, seq_len, hidden_dim]
                value = encoder_outputs
                
                attn_output, attn_weights = self.attention(query, key, value)
                
                # 计算pointer logits
                pointer_logits = self.pointer_v(torch.tanh(self.pointer_head(attn_output)))
                pointer_logits = pointer_logits.squeeze(-1)  # [batch_size, 1]
                
                # 对所有零件计算logits
                step_logits = []
                for i in range(seq_len_actual):
                    # 计算每个零件被选择的概率
                    part_query = encoder_outputs[:, i:i+1, :]  # [batch_size, 1, hidden_dim]
                    _, part_attn = self.attention(decoder_output, part_query, part_query)
                    part_logit = self.pointer_v(torch.tanh(self.pointer_head(decoder_output)))
                    
                    # 使用encoder output和attention计算logits
                    combined_features = torch.tanh(
                        self.pointer_head(decoder_output) + 
                        self.pointer_head(encoder_outputs[:, i:i+1, :])
                    )
                    part_logit = self.pointer_v(combined_features).squeeze(-1)  # [batch_size, 1]
                    step_logits.append(part_logit)
                
                step_logits = torch.cat(step_logits, dim=1)  # [batch_size, seq_len]
                all_logits.append(step_logits)
                
                # Teacher forcing: 使用真实的下一个选择作为输入
                if step < seq_len_actual - 1 and target_sequence is not None:
                    if random.random() < teacher_forcing_ratio:
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
            all_logits = self._greedy_decode(encoder_outputs, decoder_hidden, seq_len_actual)
        
        # 堆叠所有时间步的logits
        return torch.stack(all_logits, dim=1)  # [batch_size, seq_len, seq_len]

    def _greedy_decode(self, encoder_outputs, decoder_hidden, seq_len):
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
            
            all_logits.append(step_logits)
            
            # 选择最佳零件
            if step < seq_len - 1:
                next_idx = torch.argmax(step_logits, dim=1)
                # 更新mask
                selected_mask[torch.arange(batch_size), next_idx] = True
                # 更新decoder input
                decoder_input = encoder_outputs[torch.arange(batch_size), next_idx].unsqueeze(1)
        
        return all_logits

    def predict(self, parts, bin_info):
        """推理模式的预测"""
        with torch.no_grad():
            batch_size, seq_len = parts.shape[:2]
            device = parts.device
            
            # 编码输入
            encoder_outputs = self.encode_inputs(parts, bin_info)
            
            # 初始化
            decoder_hidden = (
                torch.zeros(1, batch_size, self.hidden_dim, device=device),
                torch.zeros(1, batch_size, self.hidden_dim, device=device)
            )
            
            predictions = []
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
                
                # 应用mask
                step_logits.masked_fill_(selected_mask, float('-inf'))
                
                # 选择最佳零件
                next_idx = torch.argmax(step_logits, dim=1)
                predictions.append(next_idx)
                
                # 更新状态
                if step < seq_len - 1:
                    selected_mask[torch.arange(batch_size), next_idx] = True
                    decoder_input = encoder_outputs[torch.arange(batch_size), next_idx].unsqueeze(1)
            
            return torch.stack(predictions, dim=1)  # [batch_size, seq_len]
