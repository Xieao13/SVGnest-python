import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_rectangle_vertices(width, height, x=0, y=0):
    """计算矩形顶点坐标"""
    return [
        {'x': x + width, 'y': y + height},
        {'x': x, 'y': y + height},
        {'x': x, 'y': y},
        {'x': x + width, 'y': y}
    ]


def construct_adam(arranged_parts, original_indices):
    """构建ADAM数据结构"""
    arranged_parts = arranged_parts.detach().cpu().numpy()[0]
    original_indices = original_indices.detach().cpu().numpy()
    adam = []

    for idx, (width, height) in enumerate(arranged_parts):
        if width > 0 and height > 0:
            adam.append([
                calculate_rectangle_vertices(width, height),
                {'id': int(original_indices[idx])},
                {'source': int(original_indices[idx])}
            ])

    return adam


class Attention(nn.Module):
    """自定义Attention层"""

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, values):
        # 计算注意力分数
        scores = torch.matmul(query, values.transpose(-2, -1))

        # 应用softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 加权求和得到上下文向量
        context = torch.matmul(attention_weights, values)

        return context


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(PolicyNetwork, self).__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = Attention()
        self.dense = nn.Linear(hidden_dim, seq_len)
        self.rotation_output = nn.Linear(hidden_dim, 4)  # 4种旋转角度: 0, 90, 180, 270

        # 初始化权重
        nn.init.kaiming_normal_(self.rotation_output.weight)

    def encode_decode_step(self, inputs, decoder_input, decoder_state, selected_indices=None):
        """共用的编码-解码-注意力计算步骤"""
        # 编码
        encoder_output, _ = self.encoder(inputs)
        batch_size = inputs.size(0)

        # 如果没有decoder输入，创建一个零张量
        if decoder_input is None:
            decoder_input = torch.zeros((batch_size, 1, inputs.size(2)), device=inputs.device)

        # 解码
        decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)

        # 注意力机制
        query = decoder_output.unsqueeze(1)
        attention_output = self.attention(query, encoder_output)
        attention_output = attention_output.squeeze(1)
        logits = self.dense(attention_output)

        # 处理mask - 将已选择的位置标记为负无穷
        if selected_indices is not None:
            mask = torch.zeros_like(logits)
            for idx in selected_indices:
                mask = mask + F.one_hot(idx, num_classes=self.seq_len).sum(dim=1)
            logits = torch.where(mask == 0, logits, torch.tensor(float('-inf'), device=logits.device))

        return encoder_output, logits, attention_output, decoder_state

    def forward(self, inputs, target_indices=None):
        """前向传播，支持训练和推理模式"""
        batch_size = inputs.size(0)
        decoder_input = None
        decoder_state = None

        # 使用Python列表进行收集
        all_logits = []
        all_rotation_logits = []
        selected_indices = []
        rotation_decisions = []

        for step in range(self.seq_len):
            # 编码-解码-注意力计算
            encoder_output, logits, attention_output, decoder_state = self.encode_decode_step(
                inputs, decoder_input, decoder_state,
                selected_indices if selected_indices else None
            )

            # 保存当前步骤的logits用于计算损失
            all_logits.append(logits)

            # 根据是否提供目标索引，选择预测方式
            if target_indices is not None and step < target_indices.size(1):
                # 训练模式：使用目标索引
                idx = target_indices[:, step:step + 1]
            else:
                # 推理模式：使用argmax
                idx = torch.argmax(logits, dim=1, keepdim=True)

            selected_indices.append(idx)

            # 旋转预测
            rotation_logits = self.rotation_output(attention_output)
            all_rotation_logits.append(rotation_logits)

            if target_indices is not None and step < target_indices.size(1):
                # 训练模式不需要做旋转决策，留给损失函数处理
                pass
            else:
                # 推理模式：选择最高概率的旋转
                rotation_decision = torch.argmax(rotation_logits, dim=1, keepdim=True)
                rotation_decisions.append(rotation_decision)

            # 更新decoder输入
            idx_squeezed = idx.squeeze(1)
            decoder_input = inputs[torch.arange(batch_size, device=idx.device), idx_squeezed].unsqueeze(1)

        # 合并所有结果
        all_logits = torch.stack(all_logits, dim=1)  # [batch_size, seq_len, seq_len]
        all_rotation_logits = torch.stack(all_rotation_logits, dim=1)  # [batch_size, seq_len, 4]
        selected_indices_tensor = torch.cat(selected_indices, dim=1)  # [batch_size, seq_len]

        if target_indices is None:
            rotation_decisions_tensor = torch.cat(rotation_decisions, dim=1)  # [batch_size, seq_len]
            return selected_indices_tensor, rotation_decisions_tensor, all_logits, all_rotation_logits
        else:
            return all_logits, all_rotation_logits

    def predict(self, inputs):
        """推理模式"""
        selected_indices, rotation_decisions, _, _ = self.forward(inputs)
        return selected_indices, rotation_decisions