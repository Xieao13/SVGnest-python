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
        # query: [batch_size, 1, hidden_dim]
        # values: [batch_size, seq_len, hidden_dim]
        
        batch_size = query.size(0)
        seq_len = values.size(1)
        
        # 扩展query以匹配values的序列长度
        query_expanded = query.expand(batch_size, seq_len, self.hidden_dim)
        
        # 拼接query和values
        combined = torch.cat([query_expanded, values], dim=-1)  # [batch_size, seq_len, hidden_dim * 2]
        
        # 计算注意力分数
        energy = torch.tanh(self.attention(combined))  # [batch_size, seq_len, hidden_dim]
        attention_scores = self.v(energy).squeeze(-1)  # [batch_size, seq_len]
        
        # 应用softmax得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 加权求和得到上下文向量
        context = torch.bmm(attention_weights.unsqueeze(1), values)  # [batch_size, 1, hidden_dim]
        
        return context, attention_weights


class ImprovedPolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, bin_dim=2):
        super(ImprovedPolicyNetwork, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.bin_dim = bin_dim
        
        # Bin信息编码器
        self.bin_encoder = nn.Sequential(
            nn.Linear(bin_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # 零件特征编码器
        self.part_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # LSTM编码器和解码器
        combined_dim = hidden_dim // 4 + hidden_dim // 2
        self.encoder = nn.LSTM(combined_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.decoder = nn.LSTM(combined_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        
        # 改进的注意力机制
        self.attention = Attention(hidden_dim)
        
        # 输出层
        self.dense = nn.Linear(hidden_dim, seq_len)
        self.rotation_output = nn.Linear(hidden_dim, 4)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs, bin_info, target_indices=None, teacher_forcing_ratio=1.0):
        """前向传播，支持计划抽样"""
        batch_size = inputs.size(0)
        
        # 编码bin信息
        bin_encoded = self.bin_encoder(bin_info)  # [batch_size, hidden_dim//4]
        bin_encoded = bin_encoded.unsqueeze(1).expand(-1, inputs.size(1), -1)
        
        # 编码零件信息
        parts_encoded = self.part_encoder(inputs)  # [batch_size, seq_len, hidden_dim//2]
        
        # 拼接特征
        combined_features = torch.cat([parts_encoded, bin_encoded], dim=-1)
        
        # 编码
        encoder_output, _ = self.encoder(combined_features)
        
        # 初始化解码器状态
        h0 = torch.zeros(2, batch_size, self.hidden_dim, device=inputs.device)
        c0 = torch.zeros(2, batch_size, self.hidden_dim, device=inputs.device)
        decoder_state = (h0, c0)
        
        all_logits = []
        all_rotation_logits = []
        selected_indices = []
        
        # 初始解码器输入
        decoder_input = torch.zeros(batch_size, 1, combined_features.size(-1), device=inputs.device)
        
        for step in range(self.seq_len):
            # 解码
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            
            # 注意力机制
            context, attention_weights = self.attention(decoder_output, encoder_output)
            context = context.squeeze(1)  # [batch_size, hidden_dim]
            
            # 层归一化和dropout
            context = self.layer_norm(context)
            context = self.dropout(context)
            
            # 生成logits
            logits = self.dense(context)
            
            # 改进的mask处理 - 基于实际已选择的索引
            if selected_indices:
                mask = torch.zeros_like(logits)
                for prev_idx in selected_indices:
                    mask.scatter_(1, prev_idx, 1)
                # 使用-100而不是-1e9，避免数值不稳定
                logits = logits.masked_fill(mask.bool(), -100)
            
            # 数值稳定性检查
            logits = torch.clamp(logits, min=-100, max=100)
            
            all_logits.append(logits)
            
            # 旋转预测
            rotation_logits = self.rotation_output(context)
            # 旋转logits也进行数值稳定性检查
            rotation_logits = torch.clamp(rotation_logits, min=-100, max=100)
            all_rotation_logits.append(rotation_logits)
            
            # 选择下一个索引 - 关键修复：在训练时使用真实目标更新selected_indices
            if target_indices is not None and step < target_indices.size(1):
                # 训练模式：使用计划抽样决定预测方式，但始终用真实目标更新状态
                if random.random() < teacher_forcing_ratio:
                    # 使用真实目标进行预测
                    idx = target_indices[:, step:step + 1]
                else:
                    # 使用模型预测
                    idx = torch.argmax(logits, dim=1, keepdim=True)
                
                # 关键修复：无论使用哪种预测方式，都用真实目标更新selected_indices
                # 这确保了mask逻辑与训练目标一致
                actual_idx = target_indices[:, step:step + 1]
                selected_indices.append(actual_idx)
                
                # 更新解码器输入也使用真实目标
                idx_squeezed = actual_idx.squeeze(1).clamp(0, inputs.size(1) - 1)
            else:
                # 推理模式
                idx = torch.argmax(logits, dim=1, keepdim=True)
                selected_indices.append(idx)
                idx_squeezed = idx.squeeze(1).clamp(0, inputs.size(1) - 1)
            
            # 更新解码器输入
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


def train_improved(model, train_loader, optimizer, device, epochs=20):
    """改进的训练函数"""
    model.train()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    for epoch in range(epochs):
        total_loss = 0
        total_placement_loss = 0
        total_rotation_loss = 0
        total_samples = 0
        
        # 计划抽样：逐渐减少teacher forcing
        teacher_forcing_ratio = max(0.1, 1.0 - epoch * 0.05)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            parts = batch['parts'].to(device)
            bins = batch['bins'].to(device)
            placement_targets = batch['placement_orders'].to(device)
            rotation_targets = batch['rotations'].to(device)
            valid_lengths = batch['valid_lengths'].to(device)
            
            # 调试：检查输入数据
            if batch_idx % 100 == 0:  # 每100个batch检查一次
                print(f"\n=== Batch {batch_idx} Debug Info ===")
                print(f"Parts shape: {parts.shape}, range: [{parts.min():.3f}, {parts.max():.3f}]")
                print(f"Bins shape: {bins.shape}, range: [{bins.min():.3f}, {bins.max():.3f}]")
                print(f"Placement targets range: [{placement_targets.min()}, {placement_targets.max()}]")
                print(f"Rotation targets range: [{rotation_targets.min()}, {rotation_targets.max()}]")
                print(f"Valid lengths: {valid_lengths[:5].tolist()}")
                
                # 检查是否有异常值
                if torch.isnan(parts).any():
                    print("WARNING: NaN detected in parts!")
                if torch.isnan(bins).any():
                    print("WARNING: NaN detected in bins!")
                if torch.isinf(parts).any():
                    print("WARNING: Inf detected in parts!")
                if torch.isinf(bins).any():
                    print("WARNING: Inf detected in bins!")
            
            optimizer.zero_grad()
            
            # 前向传播
            placement_logits, rotation_logits = model(
                parts, bins, placement_targets, teacher_forcing_ratio
            )
            
            # 调试：检查模型输出
            if batch_idx % 100 == 0:
                print(f"Placement logits shape: {placement_logits.shape}")
                print(f"Placement logits range: [{placement_logits.min():.3f}, {placement_logits.max():.3f}]")
                print(f"Rotation logits shape: {rotation_logits.shape}")
                print(f"Rotation logits range: [{rotation_logits.min():.3f}, {rotation_logits.max():.3f}]")
                
                # 检查logits中的异常值
                if torch.isnan(placement_logits).any():
                    print("WARNING: NaN detected in placement_logits!")
                if torch.isnan(rotation_logits).any():
                    print("WARNING: NaN detected in rotation_logits!")
                if torch.isinf(placement_logits).any():
                    print("WARNING: Inf detected in placement_logits!")
                    inf_positions = torch.isinf(placement_logits)
                    print(f"Inf positions in placement_logits: {inf_positions.sum()} out of {placement_logits.numel()}")
                if torch.isinf(rotation_logits).any():
                    print("WARNING: Inf detected in rotation_logits!")
                    inf_positions = torch.isinf(rotation_logits)
                    print(f"Inf positions in rotation_logits: {inf_positions.sum()} out of {rotation_logits.numel()}")
            
            # 计算损失
            placement_loss = 0
            rotation_loss = 0
            valid_count = 0
            
            for i in range(parts.size(0)):
                valid_len = valid_lengths[i].item()
                for j in range(valid_len):
                    if j < placement_logits.size(1) and j < rotation_logits.size(1):
                        # 放置损失
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
            
            # 检查损失是否有效
            total_loss_batch = placement_loss + rotation_loss
            
            # 调试：详细的损失检查
            if batch_idx % 100 == 0:
                print(f"Losses - Placement: {placement_loss.item():.6f}, Rotation: {rotation_loss.item():.6f}, Total: {total_loss_batch.item():.6f}")
            
            if not torch.isfinite(total_loss_batch):
                print(f"\n!!! INVALID LOSS DETECTED !!!")
                print(f"Epoch: {epoch+1}, Batch: {batch_idx}")
                print(f"Placement loss: {placement_loss.item()}")
                print(f"Rotation loss: {rotation_loss.item()}")
                print(f"Total loss: {total_loss_batch.item()}")
                print(f"Valid count: {valid_count}")
                
                # 检查模型参数
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"NaN in parameter {name}")
                    if torch.isinf(param).any():
                        print(f"Inf in parameter {name}")
                
                logger.warning(f"Invalid loss detected at epoch {epoch+1}, batch {batch_idx}: {total_loss_batch.item()}, skipping batch")
                continue
            
            # 检查损失是否过大
            if total_loss_batch.item() > 100:
                print(f"\n!!! VERY LARGE LOSS DETECTED !!!")
                print(f"Loss: {total_loss_batch.item()}, clipping to 100")
                logger.warning(f"Very large loss detected: {total_loss_batch.item()}, clipping to 100")
                total_loss_batch = torch.clamp(total_loss_batch, max=100)
            
            total_loss_batch.backward()
            
            # 检查梯度
            total_norm = 0
            nan_grad_params = []
            inf_grad_params = []
            
            for name, p in model.named_parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    
                    if torch.isnan(p.grad).any():
                        nan_grad_params.append(name)
                    if torch.isinf(p.grad).any():
                        inf_grad_params.append(name)
            
            total_norm = total_norm ** (1. / 2)
            
            if batch_idx % 100 == 0:
                print(f"Gradient norm: {total_norm:.6f}")
            
            if nan_grad_params:
                print(f"NaN gradients in: {nan_grad_params}")
            if inf_grad_params:
                print(f"Inf gradients in: {inf_grad_params}")
            
            if total_norm > 10:
                print(f"!!! LARGE GRADIENT NORM: {total_norm}")
                logger.warning(f"Large gradient norm detected: {total_norm}")
            
            # 更强的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            
            # 更新统计信息
            total_loss += total_loss_batch.item()
            total_placement_loss += placement_loss.item()
            total_rotation_loss += rotation_loss.item()
            total_samples += 1
            
            # 记录批次级别的指标到 wandb
            wandb.log({
                'batch_placement_loss': placement_loss.item(),
                'batch_rotation_loss': rotation_loss.item(),
                'batch_total_loss': total_loss_batch.item(),
                'teacher_forcing_ratio': teacher_forcing_ratio,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'gradient_norm': total_norm,
                'batch': batch_idx + epoch * len(train_loader)
            })
            
            progress_bar.set_postfix({
                'loss': total_loss_batch.item(),
                'tf_ratio': teacher_forcing_ratio,
                'grad_norm': f"{total_norm:.2f}"
            })
        
        # 计算并记录 epoch 级别的指标
        avg_loss = total_loss / total_samples
        avg_placement_loss = total_placement_loss / total_samples
        avg_rotation_loss = total_rotation_loss / total_samples
        
        scheduler.step(avg_loss)
        
        # 记录 epoch 级别的指标到 wandb
        wandb.log({
            'epoch_total_loss': avg_loss,
            'epoch_placement_loss': avg_placement_loss,
            'epoch_rotation_loss': avg_rotation_loss,
            'epoch_teacher_forcing_ratio': teacher_forcing_ratio,
            'epoch_learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch + 1
        })
        
        logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, TF Ratio = {teacher_forcing_ratio:.3f}, LR = {optimizer.param_groups[0]['lr']:.6f}")


def evaluate_improved(model, val_loader, device):
    """评估改进模型的性能"""
    model.eval()
    
    total_placement_loss = 0
    total_rotation_loss = 0
    correct_placements = 0
    correct_rotations = 0
    total_predictions = 0
    
    # 统计旋转预测分布
    rotation_predictions = Counter()
    rotation_targets = Counter()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            parts = batch['parts'].to(device)
            bins = batch['bins'].to(device)
            placement_targets = batch['placement_orders'].to(device)
            rotation_targets = batch['rotations'].to(device)
            valid_lengths = batch['valid_lengths'].to(device)

            # 前向传播
            placement_logits, rotation_logits = model(parts, bins, placement_targets)

            # 计算损失和准确率
            batch_placement_loss = 0
            batch_rotation_loss = 0
            batch_correct_placements = 0
            batch_correct_rotations = 0
            batch_predictions = 0
            
            for i in range(parts.size(0)):
                valid_len = valid_lengths[i].item()
                for j in range(valid_len):
                    if j < placement_logits.size(1) and j < rotation_logits.size(1):
                        # 计算损失
                        p_loss = F.cross_entropy(
                            placement_logits[i, j].unsqueeze(0),
                            placement_targets[i, j].unsqueeze(0)
                        )
                        total_placement_loss += p_loss.item()
                        batch_placement_loss += p_loss.item()
                        
                        r_loss = F.cross_entropy(
                            rotation_logits[i, j].unsqueeze(0),
                            rotation_targets[i, j].unsqueeze(0)
                        )
                        total_rotation_loss += r_loss.item()
                        batch_rotation_loss += r_loss.item()
                        
                        # 计算准确率
                        pred_placement = torch.argmax(placement_logits[i, j]).item()
                        true_placement = placement_targets[i, j].item()
                        if pred_placement == true_placement:
                            correct_placements += 1
                            batch_correct_placements += 1
                        
                        pred_rotation = torch.argmax(rotation_logits[i, j]).item()
                        true_rotation = rotation_targets[i, j].item()
                        if pred_rotation == true_rotation:
                            correct_rotations += 1
                            batch_correct_rotations += 1
                        
                        # 统计旋转分布
                        rotation_predictions[pred_rotation] += 1
                        rotation_targets[true_rotation] += 1
                        
                        total_predictions += 1
                        batch_predictions += 1
            
            # 记录批次级别的评估指标到 wandb
            if batch_predictions > 0:
                wandb.log({
                    'val_batch_placement_loss': batch_placement_loss / batch_predictions,
                    'val_batch_rotation_loss': batch_rotation_loss / batch_predictions,
                    'val_batch_placement_accuracy': batch_correct_placements / batch_predictions,
                    'val_batch_rotation_accuracy': batch_correct_rotations / batch_predictions,
                    'val_batch': batch_idx
                })

    # 计算平均指标
    avg_placement_loss = total_placement_loss / total_predictions if total_predictions > 0 else 0
    avg_rotation_loss = total_rotation_loss / total_predictions if total_predictions > 0 else 0
    placement_accuracy = correct_placements / total_predictions if total_predictions > 0 else 0
    rotation_accuracy = correct_rotations / total_predictions if total_predictions > 0 else 0

    # 记录最终的评估指标到 wandb
    wandb.log({
        'val_placement_loss': avg_placement_loss,
        'val_rotation_loss': avg_rotation_loss,
        'val_placement_accuracy': placement_accuracy,
        'val_rotation_accuracy': rotation_accuracy,
        'val_total_loss': avg_placement_loss + avg_rotation_loss
    })
    
    # 记录旋转分布到 wandb
    rotation_dist_log = {}
    for cls in range(4):
        pred_count = rotation_predictions.get(cls, 0)
        true_count = rotation_targets.get(cls, 0)
        pred_pct = pred_count / total_predictions * 100 if total_predictions > 0 else 0
        true_pct = true_count / total_predictions * 100 if total_predictions > 0 else 0
        
        rotation_dist_log[f'rotation_{cls*90}_pred_pct'] = pred_pct
        rotation_dist_log[f'rotation_{cls*90}_true_pct'] = true_pct
    
    wandb.log(rotation_dist_log)

    # 打印详细结果
    logger.info(f"Evaluation Results:")
    logger.info(f"  Placement Loss: {avg_placement_loss:.4f}")
    logger.info(f"  Rotation Loss: {avg_rotation_loss:.4f}")
    logger.info(f"  Placement Accuracy: {placement_accuracy:.4f}")
    logger.info(f"  Rotation Accuracy: {rotation_accuracy:.4f}")
    
    logger.info(f"Rotation Prediction Distribution:")
    for cls in range(4):
        pred_count = rotation_predictions.get(cls, 0)
        true_count = rotation_targets.get(cls, 0)
        pred_pct = pred_count / total_predictions * 100 if total_predictions > 0 else 0
        true_pct = true_count / total_predictions * 100 if total_predictions > 0 else 0
        logger.info(f"  {cls*90}度: 预测 {pred_count} ({pred_pct:.1f}%), 真实 {true_count} ({true_pct:.1f}%)")

    return {
        'placement_loss': avg_placement_loss,
        'rotation_loss': avg_rotation_loss,
        'placement_accuracy': placement_accuracy,
        'rotation_accuracy': rotation_accuracy,
        'rotation_predictions': dict(rotation_predictions),
        'rotation_targets': dict(rotation_targets)
    }


def predict_improved(model, data_loader, device):
    """使用改进模型进行预测"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Making predictions"):
            parts = batch['parts'].to(device)
            bins = batch['bins'].to(device)
            valid_lengths = batch['valid_lengths'].to(device)
            
            # 预测
            selected_indices = model.predict(parts, bins)
            
            # 转换为CPU并处理每个样本
            selected_indices = selected_indices.cpu().numpy()
            
            for i in range(len(parts)):
                valid_len = valid_lengths[i].item()
                predictions.append({
                    'bin': bins[i].cpu().numpy().tolist(),
                    'parts': parts[i][:valid_len].cpu().numpy().tolist(),
                    'predicted_placement': selected_indices[i][:valid_len].tolist(),
                    'predicted_rotation': [0] * valid_len  # 暂时设为0，需要从rotation_logits获取
                })
    
    return predictions


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 改进的设备检测
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"Using device: {device} (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        logger.info(f"Using device: {device} (GPU not available)")
    
    # 初始化 wandb
    wandb.init(
        project="bin-packing-improved",  # 项目名称
        name="improved-policy-network",  # 运行名称
        config={
            "learning_rate": 0.0005,  # 降低学习率
            "batch_size": 16,
            "hidden_dim": 256,
            "input_dim": 2,
            "bin_dim": 2,
            "epochs": 20,
            "device": str(device),
            "model_type": "ImprovedPolicyNetwork",
            "features": ["bin_encoding", "scheduled_sampling", "attention", "layer_norm"],
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "gradient_clipping": 0.5,  # 减小梯度裁剪阈值
            "scheduler": "ReduceLROnPlateau"
        },
        tags=["improved", "attention", "scheduled_sampling", "bin_encoding"]
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
        train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, drop_last=False
    )
    
    # 创建改进的模型
    input_dim = 2
    hidden_dim = 256  # 增加隐藏层维度
    max_seq_len = max(len(item['parts']) for i in range(len(dataset)) for item in [dataset[i]])
    
    logger.info(f"Maximum sequence length: {max_seq_len}")
    
    model = ImprovedPolicyNetwork(input_dim, hidden_dim, max_seq_len).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)  # 降低学习率
    
    # 记录模型信息到 wandb
    wandb.config.update({
        "max_seq_len": max_seq_len,
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
    })
    
    # 训练模型
    logger.info("开始训练改进模型...")
    train_improved(model, train_loader, optimizer, device, epochs=20)
    
    # 评估模型
    logger.info("评估改进模型...")
    metrics = evaluate_improved(model, val_loader, device)
    
    # 保存模型和结果
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    torch.save(model.state_dict(), 'models/improved_bin_packing_model.pth')
    logger.info("Improved model saved")
    
    # 保存评估结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/improved_evaluation_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation results saved to {results_file}")
    
    # 记录最终结果摘要到 wandb
    wandb.summary.update({
        "final_placement_accuracy": metrics['placement_accuracy'],
        "final_rotation_accuracy": metrics['rotation_accuracy'],
        "final_placement_loss": metrics['placement_loss'],
        "final_rotation_loss": metrics['rotation_loss'],
        "model_path": 'models/improved_bin_packing_model.pth',
        "results_file": results_file
    })
    
    # 结束 wandb 运行
    wandb.finish()
    
    return metrics


if __name__ == '__main__':
    main() 