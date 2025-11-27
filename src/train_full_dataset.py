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
import random
import wandb
import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class ImprovedPolicyNetwork(nn.Module):
    """改进的策略网络 - 用于完整数据集训练"""
    def __init__(self, input_dim, hidden_dim, seq_len, bin_dim=2):
        super(ImprovedPolicyNetwork, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.bin_dim = bin_dim
        
        # Bin信息编码器
        self.bin_encoder = nn.Sequential(
            nn.Linear(bin_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # 零件特征编码器
        self.part_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # 位置编码
        self.pos_encoder = nn.Embedding(seq_len, hidden_dim // 4)
        
        # LSTM编码器
        combined_dim = hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 4  # parts + bin + pos
        self.encoder = nn.LSTM(combined_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True, dropout=0.1)
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 输出层
        self.placement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 4)
        )
        
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

    def forward(self, inputs, bin_info, target_indices=None, teacher_forcing_ratio=1.0):
        """前向传播"""
        batch_size, seq_len_actual = inputs.shape[:2]
        
        # 编码bin信息
        bin_encoded = self.bin_encoder(bin_info)  # [batch_size, hidden_dim//4]
        bin_encoded = bin_encoded.unsqueeze(1).expand(-1, seq_len_actual, -1)
        
        # 编码零件信息
        parts_encoded = self.part_encoder(inputs)  # [batch_size, seq_len, hidden_dim//2]
        
        # 位置编码
        positions = torch.arange(seq_len_actual, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
        pos_encoded = self.pos_encoder(positions)  # [batch_size, seq_len, hidden_dim//4]
        
        # 拼接所有特征
        combined_features = torch.cat([parts_encoded, bin_encoded, pos_encoded], dim=-1)
        
        # LSTM编码
        encoder_output, _ = self.encoder(combined_features)
        encoder_output = self.layer_norm1(encoder_output)
        
        # 自注意力
        attn_output, _ = self.attention(encoder_output, encoder_output, encoder_output)
        attn_output = self.layer_norm2(attn_output + encoder_output)  # 残差连接
        
        # 生成placement和rotation logits
        placement_scores = self.placement_head(attn_output)  # [batch_size, seq_len, 1]
        placement_scores = placement_scores.squeeze(-1)  # [batch_size, seq_len]
        
        rotation_logits = self.rotation_head(attn_output)  # [batch_size, seq_len, 4]
        
        # 为每个时间步复制logits（简化的序列决策）
        all_placement_logits = []
        all_rotation_logits = []
        
        for step in range(seq_len_actual):
            # 对于每个时间步，使用相同的placement scores
            step_placement_logits = placement_scores.clone()
            step_rotation_logits = rotation_logits[:, step, :]
            
            all_placement_logits.append(step_placement_logits)
            all_rotation_logits.append(step_rotation_logits)
        
        # 堆叠结果
        all_placement_logits = torch.stack(all_placement_logits, dim=1)  # [batch_size, seq_len, seq_len]
        all_rotation_logits = torch.stack(all_rotation_logits, dim=1)     # [batch_size, seq_len, 4]
        
        return all_placement_logits, all_rotation_logits

    def predict(self, inputs, bin_info):
        """推理模式"""
        with torch.no_grad():
            placement_logits, rotation_logits = self.forward(inputs, bin_info)
            
            # 贪心选择
            batch_size, seq_len = inputs.shape[:2]
            selected_indices = []
            
            for i in range(batch_size):
                indices = torch.argsort(placement_logits[i, 0], descending=True)
                selected_indices.append(indices[:seq_len])
            
            selected_indices = torch.stack(selected_indices)
            return selected_indices


# 使用原有的数据集和collate_fn
from train_savedata import BinPackingDataset, collate_fn


def train_full_model(model, train_loader, val_loader, optimizer, device, epochs=20):
    """完整数据集训练函数"""
    model.train()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # 早停参数
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        total_placement_loss = 0
        total_rotation_loss = 0
        total_samples = 0
        
        # Teacher forcing衰减
        teacher_forcing_ratio = max(0.3, 1.0 - epoch * 0.03)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            parts = batch['parts'].to(device)
            bins = batch['bins'].to(device)
            placement_targets = batch['placement_orders'].to(device)
            rotation_targets = batch['rotations'].to(device)
            valid_lengths = batch['valid_lengths'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            placement_logits, rotation_logits = model(parts, bins, placement_targets, teacher_forcing_ratio)
            
            # 计算损失
            placement_loss = 0
            rotation_loss = 0
            valid_count = 0
            
            for i in range(parts.size(0)):
                valid_len = valid_lengths[i].item()
                for j in range(min(valid_len, placement_logits.size(1))):
                    # 检查目标值范围
                    p_target = placement_targets[i, j]
                    r_target = rotation_targets[i, j]
                    
                    if p_target >= placement_logits.size(-1) or p_target < 0:
                        continue
                    if r_target >= 4 or r_target < 0:
                        continue
                    
                    # 计算损失
                    p_loss = F.cross_entropy(
                        placement_logits[i, j].unsqueeze(0),
                        p_target.unsqueeze(0)
                    )
                    
                    r_loss = F.cross_entropy(
                        rotation_logits[i, j].unsqueeze(0),
                        r_target.unsqueeze(0)
                    )
                    
                    placement_loss += p_loss
                    rotation_loss += r_loss
                    valid_count += 1
            
            if valid_count > 0:
                placement_loss /= valid_count
                rotation_loss /= valid_count
            
            total_loss_batch = placement_loss + rotation_loss
            
            # 检查损失有效性
            if torch.isfinite(total_loss_batch) and total_loss_batch.item() < 100:
                total_loss_batch.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                # 更新统计
                total_loss += total_loss_batch.item()
                total_placement_loss += placement_loss.item()
                total_rotation_loss += rotation_loss.item()
                total_samples += 1
                
                # 记录到wandb
                wandb.log({
                    'batch_loss': total_loss_batch.item(),
                    'batch_placement_loss': placement_loss.item(),
                    'batch_rotation_loss': rotation_loss.item(),
                    'teacher_forcing_ratio': teacher_forcing_ratio,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'step': epoch * len(train_loader) + batch_idx
                })
                
                progress_bar.set_postfix({
                    'loss': f"{total_loss_batch.item():.4f}",
                    'p_loss': f"{placement_loss.item():.4f}",
                    'r_loss': f"{rotation_loss.item():.4f}",
                    'tf_ratio': f"{teacher_forcing_ratio:.3f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
        
        # 计算epoch平均损失
        if total_samples > 0:
            avg_train_loss = total_loss / total_samples
            avg_placement_loss = total_placement_loss / total_samples
            avg_rotation_loss = total_rotation_loss / total_samples
            
            # 验证阶段
            val_loss = evaluate_model(model, val_loader, device)
            
            # 记录epoch级别指标
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_placement_loss': avg_placement_loss,
                'train_rotation_loss': avg_rotation_loss,
                'val_loss': val_loss,
                'teacher_forcing_ratio': teacher_forcing_ratio,
                'epoch_lr': optimizer.param_groups[0]['lr']
            })
            
            logger.info(f"Epoch {epoch + 1}: "
                       f"Train Loss = {avg_train_loss:.4f}, "
                       f"Val Loss = {val_loss:.4f}, "
                       f"Placement = {avg_placement_loss:.4f}, "
                       f"Rotation = {avg_rotation_loss:.4f}, "
                       f"TF = {teacher_forcing_ratio:.3f}, "
                       f"LR = {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 'models/best_model.pth')
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        else:
            logger.warning(f"Epoch {epoch + 1}: No valid samples processed!")


def evaluate_model(model, val_loader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            parts = batch['parts'].to(device)
            bins = batch['bins'].to(device)
            placement_targets = batch['placement_orders'].to(device)
            rotation_targets = batch['rotations'].to(device)
            valid_lengths = batch['valid_lengths'].to(device)
            
            # 前向传播
            placement_logits, rotation_logits = model(parts, bins, placement_targets)
            
            # 计算损失
            batch_loss = 0
            valid_count = 0
            
            for i in range(parts.size(0)):
                valid_len = valid_lengths[i].item()
                for j in range(min(valid_len, placement_logits.size(1))):
                    p_target = placement_targets[i, j]
                    r_target = rotation_targets[i, j]
                    
                    if p_target >= placement_logits.size(-1) or p_target < 0:
                        continue
                    if r_target >= 4 or r_target < 0:
                        continue
                    
                    p_loss = F.cross_entropy(
                        placement_logits[i, j].unsqueeze(0),
                        p_target.unsqueeze(0)
                    )
                    
                    r_loss = F.cross_entropy(
                        rotation_logits[i, j].unsqueeze(0),
                        r_target.unsqueeze(0)
                    )
                    
                    batch_loss += (p_loss + r_loss).item()
                    valid_count += 1
            
            if valid_count > 0:
                total_loss += batch_loss / valid_count
                total_samples += 1
    
    return total_loss / total_samples if total_samples > 0 else float('inf')


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设备检测
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"Using device: {device} (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info(f"Using device: {device}")
    
    # 初始化wandb
    wandb.init(
        project="bin-packing-full",
        name=f"improved-policy-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "learning_rate": 0.001,
            "batch_size": 16,
            "hidden_dim": 256,
            "epochs": 20,
            "device": str(device),
            "dataset_size": "full_100k",
            "model_type": "ImprovedPolicyNetwork",
            "features": ["position_encoding", "multi_head_attention", "residual_connection", "layer_norm"]
        }
    )
    
    # 加载完整数据集
    data_file = '../output/placement-0412.jsonl'
    dataset = BinPackingDataset(data_file)
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    
    # 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, 
        drop_last=True, num_workers=4, pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, 
        drop_last=False, num_workers=4, pin_memory=True if device.type == 'cuda' else False
    )
    
    # 获取最大序列长度
    max_seq_len = max(len(item['parts']) for i in range(min(10000, len(dataset))) for item in [dataset[i]])
    logger.info(f"Maximum sequence length: {max_seq_len}")
    
    # 创建模型
    model = ImprovedPolicyNetwork(input_dim=2, hidden_dim=256, seq_len=max_seq_len)
    model.to(device)
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 更新wandb配置
    wandb.config.update({
        "max_seq_len": max_seq_len,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset)
    })
    
    # 创建模型保存目录
    os.makedirs('models', exist_ok=True)
    
    # 开始训练
    logger.info("开始训练完整数据集...")
    train_full_model(model, train_loader, val_loader, optimizer, device, epochs=20)
    
    # 保存最终模型
    torch.save(model.state_dict(), 'models/final_model.pth')
    logger.info("训练完成，模型已保存")
    
    # 结束wandb
    wandb.finish()


if __name__ == '__main__':
    main() 