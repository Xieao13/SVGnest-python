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
from loss.placement_only_loss import calculate_placement_loss
import argparse

# 设置日志

logger = logging.getLogger(__name__)


def train(model, train_loader, val_loader, optimizer, device, args, epochs=20):
    model.train()
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        total_accuracy = 0
        
        teacher_forcing_ratio = max(0.3, 1.0 - epoch * 0.1)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            parts = batch['parts'].to(device)
            bins = batch['bins'].to(device)
            placement_targets = batch['placement_orders'].to(device)
            valid_lengths = batch['valid_lengths'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            placement_logits = model(
                parts=parts, 
                bin_info=bins,
                target_sequence=placement_targets,
                valid_lengths=valid_lengths
            )
            
            # 计算损失
            batch_loss, batch_accuracy = calculate_placement_loss(
                placement_logits, 
                placement_targets, 
                valid_lengths
            )

            if not torch.isfinite(batch_loss):
                logger.info(f"batch_loss is infinite, skip this batch")
                # print(placement_logits.tolist())
                # print(placement_targets.tolist())
                # print(valid_lengths.tolist())
                # print(batch_loss.item())
                # print(batch_accuracy)
                break
            
            if torch.isfinite(batch_loss) and batch_loss.item() < 100:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += batch_loss.item()
                total_accuracy += batch_accuracy
                total_samples += 1
                
                # 记录到wandb
                wandb.log({
                    'batch_loss': batch_loss.item(),
                    'batch_accuracy': batch_accuracy,
                    'teacher_forcing_ratio': teacher_forcing_ratio,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'step': epoch * len(train_loader) + batch_idx
                })
                
                progress_bar.set_postfix({
                    'loss': f"{batch_loss.item():.4f}",
                    'acc': f"{batch_accuracy:.3f}",
                    'tf_ratio': f"{teacher_forcing_ratio:.3f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
        
        scheduler.step()
        
        if total_samples > 0:
            avg_train_loss = total_loss / total_samples
            avg_train_acc = total_accuracy / total_samples
            
            # 验证
            val_loss, val_acc = evaluate(model, val_loader, device)
            
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
                best_model_path = os.path.join(args.best_model_dir, 'best_improved_placement_model.pth')
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"保存新的最佳模型，验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.3f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"早停触发，在第 {epoch + 1} 轮后停止训练")
                    break


def evaluate(model, val_loader, device):
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
            
            placement_logits = model(
                parts=parts, 
                bin_info=bins,
                target_sequence=placement_targets,
                valid_lengths=valid_lengths
            )
            
            # 使用提取的损失函数
            loss, accuracy = calculate_placement_loss(
                placement_logits, 
                placement_targets, 
                valid_lengths
            )
            
            total_loss += loss.item()
            total_accuracy += accuracy
            total_samples += 1
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_acc = total_accuracy / total_samples if total_samples > 0 else 0
    return avg_loss, avg_acc


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 使用argparse解析命令行参数
    parser = argparse.ArgumentParser(description="Train the improved placement model")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loaders")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--max_seq_len", type=int, default=60, help="Maximum sequence length for the model")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer")
    
    parser.add_argument("--train_data_file", type=str, default="./data/placement-0529-ga-20epoch-norotation/train.jsonl", help="Path to the training data file")
    parser.add_argument("--test_data_file", type=str, default="./data/placement-0529-ga-20epoch-norotation/test.jsonl", help="Path to the test/validation data file")
    parser.add_argument("--best_model_dir", type=str, default="./output/models", help="Directory to save the best model")
    parser.add_argument("--final_model_dir", type=str, default="./output/models", help="Directory to save the final model")
    
    parser.add_argument("--wandb_project", type=str, default="bin-packing-placement-improved", help="WandB project name")
    parser.add_argument("--wandb_name_prefix", type=str, default="improved-placement", help="Prefix for WandB run name")
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"], help="WandB mode")
    
    args = parser.parse_args()

    # 设备检测
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"使用设备: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.info(f"使用设备: {device}")
    
    # 初始化wandb
    wandb.init(
        project=args.wandb_project,
        name=f"{args.wandb_name_prefix}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        settings=wandb.Settings(init_timeout=300),
        mode=args.wandb_mode,
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "device": str(device),
            "dataset_train": args.train_data_file,
            "dataset_val": args.test_data_file,
            "model_type": "PlacementOnlyNetwork",
            "features": ["transformer_encoder", "pointer_network", "autoregressive", "masked_attention"],
            "task": "placement_only_improved",
            "max_seq_len": args.max_seq_len,
            "weight_decay": args.weight_decay
        }
    )
    
    # 数据加载
    train_dataset = PlacementOnlyDataset(args.train_data_file)
    val_dataset = PlacementOnlyDataset(args.test_data_file) # 加载测试集作为验证集

    logger.info(f"加载训练数据集，包含 {len(train_dataset)} 个样本")
    logger.info(f"加载验证数据集，包含 {len(val_dataset)} 个样本")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=PlacementOnly_collate_fn,
        drop_last=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=PlacementOnly_collate_fn,
        drop_last=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建改进的模型
    model = PlacementOnlyNetwork(input_dim=2, hidden_dim=args.hidden_dim, seq_len=args.max_seq_len)
    model.to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 创建模型保存目录
    os.makedirs(args.best_model_dir, exist_ok=True)
    os.makedirs(args.final_model_dir, exist_ok=True)
    
    # 开始训练
    logger.info("开始训练改进的放置顺序模型...")
    train(model, train_loader, val_loader, optimizer, device, args, epochs=args.epochs)
    
    # 保存最终模型
    final_model_path = os.path.join(args.final_model_dir, 'improved_placement_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"训练完成，改进模型已保存到: {final_model_path}")
    
    wandb.finish()


if __name__ == '__main__':
    main() 