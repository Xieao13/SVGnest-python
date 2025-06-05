#!/usr/bin/env python3
"""
放置顺序模型的损失计算
"""

import torch
import torch.nn.functional as F


def calculate_placement_loss(placement_logits, placement_targets, valid_lengths):
    """
    计算放置顺序的损失和准确率
    
    Args:
        placement_logits: [batch_size, seq_len, max_parts]
        placement_targets: [batch_size, seq_len] 
        valid_lengths: [batch_size]
    
    Returns:
        loss: 平均损失
        accuracy: 准确率
    """
    total_loss = 0
    total_correct = 0
    total_valid = 0
    
    for i in range(placement_logits.size(0)):
        valid_len = valid_lengths[i].item()
        
        for step in range(valid_len):
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
                total_loss += loss
                
                # 计算准确率
                pred = torch.argmax(masked_logits[:valid_len])
                if pred == target:
                    total_correct += 1
                total_valid += 1
    
    if total_valid > 0:
        avg_loss = total_loss / total_valid
        accuracy = total_correct / total_valid
    else:
        avg_loss = torch.tensor(0.0)
        accuracy = 0.0
    
    return avg_loss, accuracy