#!/usr/bin/env python3
"""
放置顺序模型的损失计算
"""

import torch
import torch.nn.functional as F
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def calculate_placement_loss(placement_logits, placement_targets, valid_lengths):
    """
    计算放置顺序的损失和准确率
    
    Args:
        placement_logits: [batch_size, seq_len_padded, seq_len_padded]
        placement_targets: [batch_size, seq_len_padded] 
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
                if not torch.isfinite(loss):
                    logger.debug(f"loss: {loss}")
                    logger.debug(f"masked_logits: {masked_logits}")
                    logger.debug(f"target: {target}, logits: {logits}")
                    logger.debug(f"prev_selections: {prev_selections}")
                    logger.debug(f"placement_targets: {placement_targets}")
                    logger.debug(f"valid_len: {valid_len}")
                    logger.debug(f"step: {step}")
                
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

if __name__ == "__main__":
    
    placement_logits = torch.randn(1, 10, 10)
    # 不可以重复选择
    placement_targets = torch.randint(0, 10, (1, 10))
    valid_lengths = torch.tensor([10])
    loss, accuracy = calculate_placement_loss(placement_logits, placement_targets, valid_lengths)
    print(loss, accuracy)