import numpy as np
import torch
import torch.nn as nn
import os
import json
import logging
import sys
import random


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

    def forward(self, parts, bin_info, target_sequence=None, valid_lengths=None, teacher_forcing_ratio=1.0):
        """前向传播 - 真正的autoregressive预测"""
        batch_size, seq_len_padded = parts.shape[:2]
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
            for step in range(seq_len_padded):
                # 解码器一步
                decoder_output, decoder_hidden = self.decoder_rnn(decoder_input, decoder_hidden)

                # Pointer network logits计算（精简版，与_greedy_decode一致）
                combined_features = torch.tanh(
                    self.pointer_head(decoder_output) + self.pointer_head(encoder_outputs)
                )  # [batch_size, seq_len_padded, hidden_dim]
                step_logits = self.pointer_v(combined_features).squeeze(-1)  # [batch_size, seq_len_padded]
                
                # 如果提供了valid_lengths，mask掉padding部分
                if valid_lengths is not None:
                    # 创建padding mask
                    padding_mask = torch.zeros_like(step_logits, dtype=torch.bool)
                    for i, valid_len in enumerate(valid_lengths):
                        padding_mask[i, valid_len:] = True
                    step_logits.masked_fill_(padding_mask, float('-inf'))
                
                all_logits.append(step_logits)

                # Teacher forcing: 使用真实的下一个选择作为输入
                if step < seq_len_padded - 1 and target_sequence is not None:
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
                # 创建padding mask
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

    def predict(self, parts, bin_info, valid_lengths=None):
        """推理模式的预测"""
        with torch.no_grad():
            batch_size, seq_len_padded = parts.shape[:2]
            device = parts.device
            
            # 编码输入
            encoder_outputs = self.encode_inputs(parts, bin_info)
            
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


class Attention(nn.Module):
    """自定义Attention层"""

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, values):
        # query: [batch_size, 1, hidden_dim]
        # values: [batch_size, seq_len, hidden_dim]
        
        # 计算注意力分数
        scores = torch.matmul(query, values.transpose(-2, -1))  # [batch_size, 1, seq_len]
        
        # 应用softmax得到注意力权重
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        # 加权求和得到上下文向量
        context = torch.matmul(attention_weights, values)  # [batch_size, 1, hidden_dim]
        
        return context


class PlacementOnlyNoBinNetwork(nn.Module):
    """基于PolicyNetwork结构的放置顺序网络 - 不使用bin信息，删除rotation模块"""
    
    def __init__(self, input_dim, hidden_dim, seq_len, bin_dim=2):
        super(PlacementOnlyNoBinNetwork, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        # bin_dim参数保留以保持接口一致性，但不使用
        
        # LSTM编码器和解码器
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # 注意力机制
        self.attention = Attention()
        
        # 输出层
        self.dense = nn.Linear(hidden_dim, seq_len)
        
        # 层归一化（默认使用Identity）
        self.layer_norm = nn.Identity()
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.kaiming_normal_(self.dense.weight)
        
        # 其他层使用默认初始化
        for name, param in self.named_parameters():
            if 'weight' in name and 'dense' not in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=0.1)
                else:
                    nn.init.uniform_(param, -0.01, 0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def encode_decode_step(self, inputs, decoder_input, decoder_state, selected_indices=None):
        """共用的编码-解码-注意力计算步骤"""
        # 编码
        encoder_output, _ = self.encoder(inputs)  # [batch_size, seq_len, hidden_dim]
        batch_size = inputs.size(0)

        # 如果没有decoder输入，创建一个零张量
        if decoder_input is None:
            decoder_input = torch.zeros((batch_size, 1, inputs.size(2)), device=inputs.device)

        # 如果没有decoder状态，初始化一个
        if decoder_state is None:
            h0 = torch.zeros((1, batch_size, self.hidden_dim), device=inputs.device)
            c0 = torch.zeros((1, batch_size, self.hidden_dim), device=inputs.device)
            decoder_state = (h0, c0)

        # 解码
        decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)  # [batch_size, 1, hidden_dim]

        # 注意力机制
        attention_output = self.attention(decoder_output, encoder_output)  # [batch_size, 1, hidden_dim]
        attention_output = attention_output.squeeze(1)  # [batch_size, hidden_dim]
        
        # 应用层归一化
        attention_output = self.layer_norm(attention_output)
        
        # 生成logits
        logits = self.dense(attention_output)  # [batch_size, seq_len]

        # 处理mask - 将已选择的位置标记为负无穷
        if selected_indices is not None and len(selected_indices) > 0:
            mask = torch.zeros_like(logits, device=logits.device, dtype=logits.dtype)
            for idx in selected_indices:
                idx_squeezed = idx.squeeze(1).unsqueeze(1)
                ones = torch.ones_like(idx_squeezed, device=mask.device, dtype=mask.dtype)
                mask = mask.scatter_(1, idx_squeezed, ones)
            # 将已选择的位置标记为负无穷
            logits = torch.where(mask == 0, logits, torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype))

        return encoder_output, logits, attention_output, decoder_state

    def forward(self, parts, bin_info, target_sequence=None, teacher_forcing_ratio=1.0):
        """前向传播 - 保持与PlacementOnlyNetwork相同的接口"""
        # bin_info参数保留以保持接口一致性，但不使用
        batch_size, seq_len_actual = parts.shape[:2]
        
        # 使用parts作为inputs
        inputs = parts
        
        decoder_input = None
        decoder_state = None

        # 存储所有时间步的输出
        all_logits = []
        selected_indices = []

        for step in range(seq_len_actual):
            # 编码-解码-注意力计算
            encoder_output, logits, attention_output, decoder_state = self.encode_decode_step(
                inputs, decoder_input, decoder_state,
                selected_indices if selected_indices else None
            )

            # 保存当前步骤的logits
            all_logits.append(logits)

            # 根据是否提供目标序列，选择预测方式
            if target_sequence is not None and step < target_sequence.size(1):
                # 训练模式：使用teacher forcing
                if random.random() < teacher_forcing_ratio:
                    idx = target_sequence[:, step:step + 1]
                else:
                    # 使用模型预测的结果
                    idx = torch.argmax(logits, dim=1, keepdim=True)
            else:
                # 推理模式：使用argmax
                idx = torch.argmax(logits, dim=1, keepdim=True)

            selected_indices.append(idx)

            # 更新decoder输入
            if step < seq_len_actual - 1:
                idx_squeezed = idx.squeeze(1)  # [batch_size]
                
                # 确保索引在有效范围内
                idx_squeezed = idx_squeezed.clamp(0, inputs.size(1) - 1)
                
                # 使用正确的索引方式
                batch_indices = torch.arange(batch_size, device=inputs.device)
                decoder_input = inputs[batch_indices, idx_squeezed].unsqueeze(1)  # [batch_size, 1, input_dim]

        # 堆叠所有时间步的logits
        return torch.stack(all_logits, dim=1)  # [batch_size, seq_len, seq_len]

    def predict(self, parts, bin_info, valid_lengths=None):
        """推理模式的预测 - 保持与PlacementOnlyNetwork相同的接口"""
        # bin_info参数保留以保持接口一致性，但不使用
        with torch.no_grad():
            batch_size, seq_len = parts.shape[:2]
            device = parts.device
            
            inputs = parts
            decoder_input = None
            decoder_state = None

            predictions = []
            selected_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)

            for step in range(seq_len):
                # 编码-解码-注意力计算
                encoder_output, logits, attention_output, decoder_state = self.encode_decode_step(
                    inputs, decoder_input, decoder_state, None
                )

                # 应用mask
                logits.masked_fill_(selected_mask, float('-inf'))

                # 选择最佳零件
                next_idx = torch.argmax(logits, dim=1)
                predictions.append(next_idx)

                # 更新状态
                if step < seq_len - 1:
                    selected_mask[torch.arange(batch_size), next_idx] = True
                    decoder_input = inputs[torch.arange(batch_size), next_idx].unsqueeze(1)

            return torch.stack(predictions, dim=1)  # [batch_size, seq_len]

    def load_state_dict(self, state_dict, strict=True):
        """重写load_state_dict方法以处理旧版本模型"""
        # 检查是否是旧版本的模型
        if 'layer_norm.weight' not in state_dict:
            logger.info("检测到旧版本模型，使用Identity层归一化")
            # 保持使用Identity层
            pass
        else:
            logger.info("检测到新版本模型，使用LayerNorm层归一化")
            # 替换为LayerNorm层
            self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # 调用父类的load_state_dict方法
        super().load_state_dict(state_dict, strict=False)
