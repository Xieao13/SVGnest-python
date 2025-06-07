#!/usr/bin/env python3
"""
测试训练修复的脚本
"""

import torch
import numpy as np
from src.dataset.dataset import PlacementOnlyDataset, PlacementOnly_collate_fn
from src.model.placement_only_model import PlacementOnlyNetwork
from src.loss.placement_only_loss import calculate_placement_loss

def test_training_step():
    """测试单个训练步骤"""
    print("=" * 80)
    print("🔍 测试训练步骤")
    print("=" * 80)
    
    # 创建简单的测试数据
    sample1 = {
        'bin': torch.tensor([10.0, 8.0]),
        'parts': torch.tensor([[2.0, 3.0], [1.5, 2.0]]),  # 2个零件
        'placement_order': torch.tensor([0, 1]),
        'efficiency': 0.85,
        'valid_length': 2
    }
    
    sample2 = {
        'bin': torch.tensor([10.0, 8.0]),
        'parts': torch.tensor([[1.0, 1.0], [2.0, 2.0], [1.5, 1.5]]),  # 3个零件
        'placement_order': torch.tensor([1, 0, 2]),
        'efficiency': 0.92,
        'valid_length': 3
    }
    
    batch = [sample1, sample2]
    collated_batch = PlacementOnly_collate_fn(batch)
    
    print("📊 Batch数据:")
    print(f"  parts shape: {collated_batch['parts'].shape}")
    print(f"  placement_orders: {collated_batch['placement_orders']}")
    print(f"  valid_lengths: {collated_batch['valid_lengths']}")
    
    # 创建模型和优化器
    model = PlacementOnlyNetwork(input_dim=2, hidden_dim=32, seq_len=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    model.train()
    
    # 模拟训练步骤
    parts = collated_batch['parts']
    bins = collated_batch['bins']
    placement_targets = collated_batch['placement_orders']
    valid_lengths = collated_batch['valid_lengths']
    
    print(f"\n🚀 开始训练步骤...")
    
    # 前向传播
    placement_logits = model(
        parts=parts, 
        bin_info=bins,
        target_sequence=placement_targets,
        valid_lengths=valid_lengths
    )
    
    print(f"  placement_logits shape: {placement_logits.shape}")
    print(f"  placement_logits统计:")
    print(f"    min: {placement_logits.min().item():.4f}")
    print(f"    max: {placement_logits.max().item():.4f}")
    print(f"    包含inf: {torch.isinf(placement_logits).any()}")
    print(f"    包含nan: {torch.isnan(placement_logits).any()}")
    
    # 计算损失
    batch_loss, batch_accuracy = calculate_placement_loss(
        placement_logits, 
        placement_targets, 
        valid_lengths
    )
    
    print(f"\n💰 损失计算:")
    print(f"  batch_loss: {batch_loss}")
    print(f"  batch_accuracy: {batch_accuracy}")
    print(f"  loss是否为inf: {torch.isinf(batch_loss)}")
    print(f"  loss是否为nan: {torch.isnan(batch_loss)}")
    print(f"  loss是否有限: {torch.isfinite(batch_loss)}")
    
    # 反向传播
    if torch.isfinite(batch_loss) and batch_loss.item() < 100:
        print(f"\n⬅️ 执行反向传播...")
        optimizer.zero_grad()
        batch_loss.backward()
        
        # 检查梯度
        total_grad_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2)
                total_grad_norm += grad_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        print(f"  总梯度范数: {total_grad_norm:.6f}")
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        print(f"  ✅ 训练步骤完成")
    else:
        print(f"  ❌ 跳过异常batch: loss={batch_loss}, finite={torch.isfinite(batch_loss)}")

def test_multiple_batches():
    """测试多个batch的处理"""
    print("\n" + "=" * 80)
    print("🔄 测试多个batch")
    print("=" * 80)
    
    # 创建多个不同的batch
    batches = []
    
    for i in range(3):
        sample1 = {
            'bin': torch.tensor([10.0, 8.0]),
            'parts': torch.tensor([[2.0+i*0.1, 3.0], [1.5, 2.0+i*0.1]]),
            'placement_order': torch.tensor([0, 1]),
            'efficiency': 0.85,
            'valid_length': 2
        }
        
        sample2 = {
            'bin': torch.tensor([10.0, 8.0]),
            'parts': torch.tensor([[1.0, 1.0+i*0.1], [2.0, 2.0], [1.5+i*0.1, 1.5]]),
            'placement_order': torch.tensor([1, 0, 2]),
            'efficiency': 0.92,
            'valid_length': 3
        }
        
        batch = [sample1, sample2]
        collated_batch = PlacementOnly_collate_fn(batch)
        batches.append(collated_batch)
    
    # 创建模型
    model = PlacementOnlyNetwork(input_dim=2, hidden_dim=32, seq_len=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.train()
    
    total_loss = 0
    total_accuracy = 0
    total_samples = 0
    
    print(f"处理 {len(batches)} 个batch...")
    
    for batch_idx, batch in enumerate(batches):
        print(f"\nBatch {batch_idx + 1}:")
        
        # 前向传播
        placement_logits = model(
            parts=batch['parts'], 
            bin_info=batch['bins'],
            target_sequence=batch['placement_orders'],
            valid_lengths=batch['valid_lengths']
        )
        
        # 计算损失
        batch_loss, batch_accuracy = calculate_placement_loss(
            placement_logits, 
            batch['placement_orders'], 
            batch['valid_lengths']
        )
        
        print(f"  loss: {batch_loss:.4f}, accuracy: {batch_accuracy:.3f}")
        
        if torch.isfinite(batch_loss) and batch_loss.item() < 100:
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += batch_loss.item()
            total_accuracy += batch_accuracy
            total_samples += 1
            print(f"  ✅ 训练成功")
        else:
            print(f"  ❌ 跳过异常batch")
    
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        print(f"\n📊 总结:")
        print(f"  处理的batch数: {total_samples}/{len(batches)}")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  平均准确率: {avg_accuracy:.3f}")
    else:
        print(f"\n❌ 没有成功处理任何batch")

if __name__ == "__main__":
    test_training_step()
    test_multiple_batches() 