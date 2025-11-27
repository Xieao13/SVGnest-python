import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import logging
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb
import argparse
import datetime
import random
from collections import Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """改进的Attention层"""

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, values):
        batch_size = query.size(0)
        seq_len = values.size(1)
        
        query_expanded = query.expand(batch_size, seq_len, self.hidden_dim)
        combined = torch.cat([query_expanded, values], dim=-1)
        
        energy = torch.tanh(self.attention(combined))
        attention_scores = self.v(energy).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        context = torch.bmm(attention_weights.unsqueeze(1), values)
        
        return context, attention_weights


class ImprovedPolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, bin_dim=2):
        super(ImprovedPolicyNetwork, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.bin_dim = bin_dim
        
        # 更强的Bin信息编码器
        self.bin_encoder = nn.Sequential(
            nn.Linear(bin_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # 更强的零件特征编码器
        self.part_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 位置编码
        self.pos_encoder = nn.Embedding(seq_len, hidden_dim)
        
        # LSTM编码器和解码器
        combined_dim = hidden_dim + hidden_dim // 2
        self.encoder = nn.LSTM(combined_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.decoder = nn.LSTM(combined_dim + hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
        # 注意力机制
        self.attention = Attention(hidden_dim)
        
        # 输出层 - 添加中间层
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, seq_len)
        )
        
        self.rotation_output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4)
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=1.0)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs, bin_info, target_indices=None, teacher_forcing_ratio=1.0):
        """改进的前向传播"""
        batch_size = inputs.size(0)
        
        # 编码bin信息
        bin_encoded = self.bin_encoder(bin_info)
        bin_encoded = bin_encoded.unsqueeze(1).expand(-1, inputs.size(1), -1)
        
        # 编码零件信息
        parts_encoded = self.part_encoder(inputs)
        
        # 添加位置编码
        positions = torch.arange(inputs.size(1), device=inputs.device).unsqueeze(0).expand(batch_size, -1)
        pos_encoded = self.pos_encoder(positions)
        parts_encoded = parts_encoded + pos_encoded
        
        # 拼接特征
        combined_features = torch.cat([parts_encoded, bin_encoded], dim=-1)
        
        # 编码
        encoder_output, _ = self.encoder(combined_features)
        encoder_output = self.layer_norm1(encoder_output)
        
        # 初始化解码器状态
        h0 = torch.zeros(2, batch_size, self.hidden_dim, device=inputs.device)
        c0 = torch.zeros(2, batch_size, self.hidden_dim, device=inputs.device)
        decoder_state = (h0, c0)
        
        all_logits = []
        all_rotation_logits = []
        selected_indices = []
        
        # 初始解码器输入
        decoder_input = torch.zeros(batch_size, 1, combined_features.size(-1), device=inputs.device)
        
        # 用于跟踪已选择的索引 (更有效的mask机制)
        selected_mask = torch.zeros(batch_size, self.seq_len, device=inputs.device)
        
        for step in range(self.seq_len):
            # 添加上下文信息到解码器输入
            if step > 0:
                # 使用平均编码器输出作为上下文
                context = encoder_output.mean(dim=1, keepdim=True)
                decoder_input_with_context = torch.cat([decoder_input, context], dim=-1)
            else:
                context = torch.zeros(batch_size, 1, self.hidden_dim, device=inputs.device)
                decoder_input_with_context = torch.cat([decoder_input, context], dim=-1)
            
            # 解码
            decoder_output, decoder_state = self.decoder(decoder_input_with_context, decoder_state)
            decoder_output = self.layer_norm2(decoder_output)
            
            # 注意力机制
            context, attention_weights = self.attention(decoder_output, encoder_output)
            context = context.squeeze(1)
            
            # 合并特征
            combined_context = torch.cat([decoder_output.squeeze(1), context], dim=-1)
            
            # 生成logits
            logits = self.dense(combined_context)
            
            # 改进的mask机制 - 使用更稳定的方法
            logits = logits + selected_mask * (-1e9)  # 使用加法而不是masked_fill
            
            # 确保数值稳定性
            logits = torch.clamp(logits, min=-1e9, max=1e9)
            
            all_logits.append(logits)
            
            # 旋转预测
            rotation_logits = self.rotation_output(combined_context)
            rotation_logits = torch.clamp(rotation_logits, min=-1e9, max=1e9)
            all_rotation_logits.append(rotation_logits)
            
            # 选择下一个索引
            if target_indices is not None and step < target_indices.size(1):
                # 训练模式：使用更灵活的teacher forcing
                if random.random() < teacher_forcing_ratio:
                    idx = target_indices[:, step:step + 1]
                else:
                    idx = torch.argmax(logits, dim=1, keepdim=True)
                # 无论如何都使用真实目标更新mask
                real_idx = target_indices[:, step:step + 1]
                batch_indices = torch.arange(batch_size, device=inputs.device).unsqueeze(1)
                selected_mask[batch_indices, real_idx] = 1
            else:
                # 推理模式
                idx = torch.argmax(logits, dim=1, keepdim=True)
                batch_indices = torch.arange(batch_size, device=inputs.device).unsqueeze(1)
                selected_mask[batch_indices, idx] = 1
            
            selected_indices.append(idx)
            
            # 更新解码器输入
            idx_squeezed = idx.squeeze(1).clamp(0, inputs.size(1) - 1)
            batch_indices = torch.arange(batch_size, device=inputs.device)
            decoder_input = combined_features[batch_indices, idx_squeezed].unsqueeze(1)
        
        # 合并结果
        all_logits = torch.stack(all_logits, dim=1)
        all_rotation_logits = torch.stack(all_rotation_logits, dim=1)
        selected_indices_tensor = torch.cat(selected_indices, dim=1)
        
        if target_indices is None:
            return selected_indices_tensor, None, all_logits, all_rotation_logits
        else:
            return all_logits, all_rotation_logits

    def predict(self, inputs, bin_info):
        """推理模式"""
        selected_indices, _, _, _ = self.forward(inputs, bin_info)
        return selected_indices


# 使用原有的数据集和collate_fn
from train_savedata import BinPackingDataset, collate_fn


def train_improved(model, train_loader, optimizer, device, epochs=30):
    """优化的训练函数"""
    model.train()
    
    # 更好的学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # 用于早停的变量
    best_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    for epoch in range(epochs):
        total_loss = 0
        total_placement_loss = 0
        total_rotation_loss = 0
        total_samples = 0
        
        # 更缓慢的teacher forcing衰减
        teacher_forcing_ratio = max(0.3, 1.0 - epoch * 0.02)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            parts = batch['parts'].to(device)
            bins = batch['bins'].to(device)
            placement_targets = batch['placement_orders'].to(device)
            rotation_targets = batch['rotations'].to(device)
            valid_lengths = batch['valid_lengths'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            placement_logits, rotation_logits = model(
                parts, bins, placement_targets, teacher_forcing_ratio
            )
            
            # 修复的损失计算 - 回到标准交叉熵
            placement_loss = 0
            rotation_loss = 0
            valid_count = 0
            
            for i in range(parts.size(0)):
                valid_len = valid_lengths[i].item()
                for j in range(min(valid_len, placement_logits.size(1))):
                    # 标准交叉熵损失 - 更稳定
                    p_loss = F.cross_entropy(
                        placement_logits[i, j].unsqueeze(0),
                        placement_targets[i, j].unsqueeze(0)
                    )
                    placement_loss += p_loss
                    
                    # 旋转损失
                    r_loss = F.cross_entropy(
                        rotation_logits[i, j].unsqueeze(0),
                        rotation_targets[i, j].unsqueeze(0)
                    )
                    rotation_loss += r_loss
                    
                    valid_count += 1
            
            if valid_count > 0:
                placement_loss /= valid_count
                rotation_loss /= valid_count
            
            # 总损失 - 简单相加
            total_loss_batch = placement_loss + rotation_loss
            
            # 检查损失是否有效
            if torch.isfinite(total_loss_batch) and total_loss_batch.item() < 100:
                total_loss_batch.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                # 更新统计信息
                total_loss += total_loss_batch.item()
                total_placement_loss += placement_loss.item()
                total_rotation_loss += rotation_loss.item()
                total_samples += 1
                
                # 记录到 wandb
                wandb.log({
                    'batch_placement_loss': placement_loss.item(),
                    'batch_rotation_loss': rotation_loss.item(),
                    'batch_total_loss': total_loss_batch.item(),
                    'teacher_forcing_ratio': teacher_forcing_ratio,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'batch': batch_idx + epoch * len(train_loader)
                })
                
                progress_bar.set_postfix({
                    'loss': f"{total_loss_batch.item():.4f}",
                    'tf_ratio': f"{teacher_forcing_ratio:.2f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
            else:
                # 跳过异常的batch
                logger.warning(f"Skipping batch with invalid loss: {total_loss_batch.item() if torch.isfinite(total_loss_batch) else 'inf/nan'}")
                continue
        
        # 计算epoch平均损失
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        avg_placement_loss = total_placement_loss / total_samples if total_samples > 0 else 0
        avg_rotation_loss = total_rotation_loss / total_samples if total_samples > 0 else 0
        
        # 记录epoch级别的指标
        wandb.log({
            'epoch_total_loss': avg_loss,
            'epoch_placement_loss': avg_placement_loss,
            'epoch_rotation_loss': avg_rotation_loss,
            'epoch_teacher_forcing_ratio': teacher_forcing_ratio,
            'epoch_learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch + 1
        })
        
        logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, "
                   f"Placement = {avg_placement_loss:.4f}, "
                   f"Rotation = {avg_rotation_loss:.4f}, "
                   f"TF = {teacher_forcing_ratio:.3f}, "
                   f"LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'models/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设备检测
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     logger.info(f"Using device: {device} (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        logger.info(f"Using device: {device}")
    
    # 初始化 wandb
    wandb.init(
        project="bin-packing-optimized",
        name="optimized-policy-network-fixed",
        config={
            "learning_rate": 0.001,  # 降低学习率
            "batch_size": 32,       
            "hidden_dim": 512,      
            "input_dim": 2,
            "bin_dim": 2,
            "epochs": 30,
            "device": str(device),
            "model_type": "OptimizedPolicyNetwork",
            "features": ["position_encoding", "cosine_annealing", "early_stopping"],
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "gradient_clipping": 1.0,
            "scheduler": "CosineAnnealingWarmRestarts"
        },
        tags=["optimized", "fixed", "position_encoding"]
    )
    
    # 加载数据
    data_file = '../output/placement-0412.jsonl'
    dataset = BinPackingDataset(data_file)
    
    # 分割数据
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, 
        drop_last=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, 
        drop_last=False, num_workers=4
    )
    
    # 创建模型
    input_dim = 2
    hidden_dim = 512
    max_seq_len = max(len(item['parts']) for i in range(len(dataset)) for item in [dataset[i]])
    
    logger.info(f"Maximum sequence length: {max_seq_len}")
    
    model = ImprovedPolicyNetwork(input_dim, hidden_dim, max_seq_len).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # 降低学习率
    
    # 记录模型信息
    wandb.config.update({
        "max_seq_len": max_seq_len,
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
    })
    
    # 训练模型
    logger.info("开始优化训练...")
    os.makedirs('models', exist_ok=True)
    train_improved(model, train_loader, optimizer, device, epochs=30)
    
    # 保存最终模型
    torch.save(model.state_dict(), 'models/final_optimized_model.pth')
    logger.info("优化模型训练完成并保存")
    
    wandb.finish()


if __name__ == '__main__':
    main() 