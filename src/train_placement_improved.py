#!/usr/bin/env python3
"""
改进的仅预测放置顺序的训练脚本
解决原有模型的架构问题，实现真正的autoregressive预测
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import PlacementOnlyDataset, PlacementOnly_collate_fn
from model.placement_only_model import PlacementOnlyNetwork
import os
import json
import logging
import sys
from tqdm import tqdm
import random
import wandb
import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def train_improved_model(model, train_loader, val_loader, optimizer, device, epochs=20):
    """训练改进的模型"""
    model.train()
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        total_accuracy = 0
        
        teacher_forcing_ratio = max(0.3, 1.0 - epoch * 0.03)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            parts = batch['parts'].to(device)
            bins = batch['bins'].to(device)
            placement_targets = batch['placement_orders'].to(device)
            valid_lengths = batch['valid_lengths'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            placement_logits = model(parts, bins, placement_targets, teacher_forcing_ratio)
            
            # 计算损失 - 使用masked cross entropy
            total_loss_batch = 0
            total_correct = 0
            total_valid = 0
            
            for i in range(parts.size(0)):
                valid_len = valid_lengths[i].item()
                
                # 为每个样本创建mask
                for step in range(valid_len):
                    # 当前步骤的目标
                    target = placement_targets[i, step]
                    logits = placement_logits[i, step]
                    
                    # 创建mask - 之前选择的零件不能再选
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    if step > 0:
                        prev_selections = placement_targets[i, :step]
                        mask[prev_selections] = True
                    
                    # 应用mask
                    masked_logits = logits.clone()
                    masked_logits[mask] = float('-inf')
                    
                    # 计算损失
                    if target < valid_len:
                        loss = F.cross_entropy(masked_logits[:valid_len].unsqueeze(0), target.unsqueeze(0))
                        total_loss_batch += loss
                        
                        # 计算准确率
                        pred = torch.argmax(masked_logits[:valid_len])
                        if pred == target:
                            total_correct += 1
                        total_valid += 1
            
            if total_valid > 0:
                total_loss_batch /= total_valid
                batch_accuracy = total_correct / total_valid
            else:
                continue
            
            if torch.isfinite(total_loss_batch) and total_loss_batch.item() < 100:
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                total_accuracy += batch_accuracy
                total_samples += 1
                
                # 记录到wandb
                wandb.log({
                    'batch_loss': total_loss_batch.item(),
                    'batch_accuracy': batch_accuracy,
                    'teacher_forcing_ratio': teacher_forcing_ratio,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'step': epoch * len(train_loader) + batch_idx
                })
                
                progress_bar.set_postfix({
                    'loss': f"{total_loss_batch.item():.4f}",
                    'acc': f"{batch_accuracy:.3f}",
                    'tf_ratio': f"{teacher_forcing_ratio:.3f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
        
        scheduler.step()
        
        if total_samples > 0:
            avg_train_loss = total_loss / total_samples
            avg_train_acc = total_accuracy / total_samples
            
            # 验证
            val_loss, val_acc = evaluate_improved_model(model, val_loader, device)
            
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': avg_train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'teacher_forcing_ratio': teacher_forcing_ratio,
                'epoch_lr': optimizer.param_groups[0]['lr']
            })
            
            logger.info(f"Epoch {epoch + 1}: "
                       f"Train Loss = {avg_train_loss:.4f}, Train Acc = {avg_train_acc:.3f}, "
                       f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.3f}, "
                       f"TF = {teacher_forcing_ratio:.3f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'models/best_improved_placement_model.pth')
                logger.info(f"保存新的最佳模型，验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.3f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"早停触发，在第 {epoch + 1} 轮后停止训练")
                    break


def evaluate_improved_model(model, val_loader, device):
    """评估改进的模型"""
    model.eval()
    total_loss = 0
    total_samples = 0
    total_accuracy = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="评估中", leave=False):
            parts = batch['parts'].to(device)
            bins = batch['bins'].to(device)
            placement_targets = batch['placement_orders'].to(device)
            valid_lengths = batch['valid_lengths'].to(device)
            
            placement_logits = model(parts, bins, placement_targets)
            
            batch_loss = 0
            batch_correct = 0
            batch_valid = 0
            
            for i in range(parts.size(0)):
                valid_len = valid_lengths[i].item()
                
                for step in range(valid_len):
                    target = placement_targets[i, step]
                    logits = placement_logits[i, step]
                    
                    # 创建mask
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    if step > 0:
                        prev_selections = placement_targets[i, :step]
                        mask[prev_selections] = True
                    
                    masked_logits = logits.clone()
                    masked_logits[mask] = float('-inf')
                    
                    if target < valid_len:
                        loss = F.cross_entropy(masked_logits[:valid_len].unsqueeze(0), target.unsqueeze(0))
                        batch_loss += loss.item()
                        
                        pred = torch.argmax(masked_logits[:valid_len])
                        if pred == target:
                            batch_correct += 1
                        batch_valid += 1
            
            if batch_valid > 0:
                total_loss += batch_loss / batch_valid
                total_accuracy += batch_correct / batch_valid
                total_samples += 1
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_acc = total_accuracy / total_samples if total_samples > 0 else 0
    return avg_loss, avg_acc


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设备检测
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"使用设备: {device} ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"使用设备: {device} (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info(f"使用设备: {device}")
    
    # 初始化wandb
    wandb.init(
        project="bin-packing-placement-improved",
        name=f"improved-placement-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        settings=wandb.Settings(init_timeout=300),
        mode="offline",
        config={
            "learning_rate": 0.001,
            "batch_size": 64,  # 减小batch size因为模型更复杂
            "hidden_dim": 256,  # 适中的hidden size
            "epochs": 25,
            "device": str(device),
            "dataset": "placement_0529.jsonl",
            "model_type": "PlacementOnlyNetwork",
            "features": ["transformer_encoder", "pointer_network", "autoregressive", "masked_attention"],
            "task": "placement_only_improved"
        }
    )
    
    # 数据加载
    data_file = '../output/placement_0529.jsonl'
    dataset = PlacementOnlyDataset(data_file)
    logger.info(f"加载数据集，包含 {len(dataset)} 个样本")
    
    # 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"训练数据集大小: {len(train_dataset)}")
    logger.info(f"验证数据集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=PlacementOnly_collate_fn,
        drop_last=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, collate_fn=PlacementOnly_collate_fn,
        drop_last=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建改进的模型
    max_seq_len = 60
    model = PlacementOnlyNetwork(input_dim=2, hidden_dim=256, seq_len=max_seq_len)
    model.to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数数量: {total_params:,}")
    logger.info(f"可训练参数数量: {trainable_params:,}")
    
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
    logger.info("开始训练改进的放置顺序模型...")
    train_improved_model(model, train_loader, val_loader, optimizer, device, epochs=25)
    
    # 保存最终模型
    torch.save(model.state_dict(), 'models/final_improved_placement_model.pth')
    logger.info("训练完成，改进模型已保存")
    
    wandb.finish()


if __name__ == '__main__':
    main() 