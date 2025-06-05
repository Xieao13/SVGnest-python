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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 使用原有的数据集和collate_fn
from train_savedata import BinPackingDataset, collate_fn


def debug_data_loading():
    """调试数据加载过程"""
    print("=" * 60)
    print("🔍 开始调试数据加载过程...")
    print("=" * 60)
    
    # 加载数据
    data_file = '../output/placement-0412.jsonl'
    dataset = BinPackingDataset(data_file)
    
    print(f"✅ 数据集大小: {len(dataset)}")
    
    # 检查前几个样本
    print("\n📊 检查前3个样本的数据:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\n样本 {i}:")
        print(f"  bin shape: {sample['bin'].shape}, values: {sample['bin'].tolist()}")
        print(f"  parts shape: {sample['parts'].shape}")
        print(f"  placement_order shape: {sample['placement_order'].shape}")
        print(f"  placement_order values: {sample['placement_order'].tolist()}")
        print(f"  rotation shape: {sample['rotation'].shape}")
        print(f"  rotation values: {sample['rotation'].tolist()}")
        print(f"  valid_length: {sample['valid_length']}")
        print(f"  efficiency: {sample['efficiency']:.4f}")
        
        # 检查placement_order是否合理
        parts_count = len(sample['parts'])
        placement_order = sample['placement_order'].tolist()
        print(f"  零件数量: {parts_count}")
        print(f"  placement_order范围: [{min(placement_order)}, {max(placement_order)}]")
        
        # 检查是否为连续索引
        expected_range = set(range(parts_count))
        actual_range = set(placement_order)
        if expected_range != actual_range:
            print(f"  ⚠️ 警告: placement_order不是连续索引!")
            print(f"     期望: {sorted(expected_range)}")
            print(f"     实际: {sorted(actual_range)}")
        else:
            print(f"  ✅ placement_order是连续索引")
    
    # 检查数据加载器
    print(f"\n🔄 检查DataLoader...")
    train_loader = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, drop_last=True
    )
    
    # 获取一个batch
    batch = next(iter(train_loader))
    print(f"\n📦 Batch数据形状:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype: {value.dtype}")
            if key == 'placement_orders':
                print(f"    范围: [{value.min().item()}, {value.max().item()}]")
                print(f"    前两个样本: {value[:2].tolist()}")
            elif key == 'rotations':
                print(f"    范围: [{value.min().item()}, {value.max().item()}]")
                print(f"    前两个样本: {value[:2].tolist()}")
            elif key == 'valid_lengths':
                print(f"    值: {value.tolist()}")
        else:
            print(f"  {key}: {type(value)}")
    
    # 检查最大序列长度
    max_seq_len = max(len(item['parts']) for i in range(min(1000, len(dataset))) for item in [dataset[i]])
    print(f"\n📏 前1000个样本的最大序列长度: {max_seq_len}")
    
    return dataset, train_loader, max_seq_len


def debug_model_forward():
    """调试模型前向传播"""
    print("\n" + "=" * 60)
    print("🔍 开始调试模型前向传播...")
    print("=" * 60)
    
    # 加载数据
    dataset, train_loader, max_seq_len = debug_data_loading()
    
    # 创建一个简化的模型用于测试
    class SimpleTestNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim, seq_len, bin_dim=2):
            super(SimpleTestNetwork, self).__init__()
            self.seq_len = seq_len
            self.hidden_dim = hidden_dim
            
            # 简化的编码器
            self.bin_encoder = nn.Linear(bin_dim, 32)
            self.part_encoder = nn.Linear(input_dim, 64)
            
            # 简化的输出层
            self.dense = nn.Linear(96, seq_len)  # 64 + 32
            self.rotation_output = nn.Linear(96, 4)
            
        def forward(self, inputs, bin_info, target_indices=None):
            batch_size, seq_len_actual = inputs.shape[:2]
            
            print(f"  输入形状 - inputs: {inputs.shape}, bin_info: {bin_info.shape}")
            
            # 编码
            bin_encoded = self.bin_encoder(bin_info)  # [batch_size, 32]
            bin_encoded = bin_encoded.unsqueeze(1).expand(-1, seq_len_actual, -1)  # [batch_size, seq_len, 32]
            
            parts_encoded = self.part_encoder(inputs)  # [batch_size, seq_len, 64]
            
            print(f"  编码后形状 - parts_encoded: {parts_encoded.shape}, bin_encoded: {bin_encoded.shape}")
            
            # 拼接
            combined = torch.cat([parts_encoded, bin_encoded], dim=-1)  # [batch_size, seq_len, 96]
            
            print(f"  拼接后形状: {combined.shape}")
            
            # 简单的平均池化
            pooled = combined.mean(dim=1)  # [batch_size, 96]
            
            # 输出
            placement_logits = self.dense(pooled)  # [batch_size, seq_len]
            rotation_logits = self.rotation_output(pooled)  # [batch_size, 4]
            
            print(f"  输出形状 - placement_logits: {placement_logits.shape}, rotation_logits: {rotation_logits.shape}")
            
            # 检查logits的数值范围
            print(f"  placement_logits范围: [{placement_logits.min().item():.3f}, {placement_logits.max().item():.3f}]")
            print(f"  rotation_logits范围: [{rotation_logits.min().item():.3f}, {rotation_logits.max().item():.3f}]")
            
            # 模拟序列输出 (为了兼容原来的训练代码)
            placement_logits_seq = placement_logits.unsqueeze(1).expand(-1, seq_len_actual, -1)
            rotation_logits_seq = rotation_logits.unsqueeze(1).expand(-1, seq_len_actual, -1)
            
            return placement_logits_seq, rotation_logits_seq
    
    # 设备
    device = torch.device("cpu")  # 先用CPU调试
    
    # 创建模型
    model = SimpleTestNetwork(input_dim=2, hidden_dim=128, seq_len=max_seq_len)
    model.to(device)
    
    print(f"\n🤖 创建简化测试模型, seq_len={max_seq_len}")
    
    # 测试一个batch
    batch = next(iter(train_loader))
    parts = batch['parts'].to(device)
    bins = batch['bins'].to(device)
    placement_targets = batch['placement_orders'].to(device)
    rotation_targets = batch['rotations'].to(device)
    valid_lengths = batch['valid_lengths'].to(device)
    
    print(f"\n🔄 测试前向传播...")
    try:
        with torch.no_grad():
            placement_logits, rotation_logits = model(parts, bins, placement_targets)
            print(f"✅ 前向传播成功!")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试损失计算
    print(f"\n📊 测试损失计算...")
    try:
        total_placement_loss = 0
        total_rotation_loss = 0
        valid_count = 0
        
        for i in range(parts.size(0)):
            valid_len = valid_lengths[i].item()
            print(f"  样本 {i}: valid_len={valid_len}")
            
            for j in range(min(valid_len, placement_logits.size(1))):
                # 检查目标值是否在合理范围内
                p_target = placement_targets[i, j].item()
                r_target = rotation_targets[i, j].item()
                
                print(f"    位置 {j}: p_target={p_target}, r_target={r_target}")
                
                if p_target >= placement_logits.size(-1):
                    print(f"    ⚠️ 警告: placement_target {p_target} 超出logits范围 {placement_logits.size(-1)}")
                    continue
                
                if r_target >= 4:
                    print(f"    ⚠️ 警告: rotation_target {r_target} 超出范围 [0,3]")
                    continue
                
                # 计算损失
                p_loss = F.cross_entropy(
                    placement_logits[i, j].unsqueeze(0),
                    placement_targets[i, j].unsqueeze(0)
                )
                
                r_loss = F.cross_entropy(
                    rotation_logits[i, j].unsqueeze(0),
                    rotation_targets[i, j].unsqueeze(0)
                )
                
                total_placement_loss += p_loss.item()
                total_rotation_loss += r_loss.item()
                valid_count += 1
                
                if j < 3:  # 只打印前3个位置的详细信息
                    print(f"    位置 {j}: p_loss={p_loss.item():.4f}, r_loss={r_loss.item():.4f}")
        
        if valid_count > 0:
            avg_p_loss = total_placement_loss / valid_count
            avg_r_loss = total_rotation_loss / valid_count
            total_loss = avg_p_loss + avg_r_loss
            
            print(f"\n📈 损失统计:")
            print(f"  平均placement_loss: {avg_p_loss:.4f}")
            print(f"  平均rotation_loss: {avg_r_loss:.4f}")
            print(f"  总损失: {total_loss:.4f}")
            print(f"  有效样本数: {valid_count}")
            
            if total_loss > 100:
                print(f"⚠️ 警告: 损失值过大!")
            else:
                print(f"✅ 损失值在合理范围内")
        
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()


def debug_data_statistics():
    """分析数据集的统计特性"""
    print("\n" + "=" * 60)
    print("📈 数据集统计分析...")
    print("=" * 60)
    
    # 加载数据
    data_file = '../output/placement-0412.jsonl'
    dataset = BinPackingDataset(data_file)
    
    # 统计信息
    seq_lengths = []
    placement_ranges = []
    rotation_ranges = []
    efficiencies = []
    bin_sizes = []
    
    print("🔄 分析前1000个样本...")
    for i in range(min(1000, len(dataset))):
        sample = dataset[i]
        
        seq_len = sample['valid_length']
        seq_lengths.append(seq_len)
        
        placement_order = sample['placement_order'].tolist()
        placement_ranges.append([min(placement_order), max(placement_order)])
        
        rotation = sample['rotation'].tolist()
        rotation_ranges.append([min(rotation), max(rotation)])
        
        efficiencies.append(sample['efficiency'])
        
        bin_size = sample['bin'].tolist()
        bin_sizes.append(bin_size)
    
    print(f"\n📊 统计结果:")
    print(f"  序列长度: 最小={min(seq_lengths)}, 最大={max(seq_lengths)}, 平均={np.mean(seq_lengths):.2f}")
    
    print(f"  placement_order范围:")
    min_vals = [r[0] for r in placement_ranges]
    max_vals = [r[1] for r in placement_ranges]
    print(f"    最小值范围: [{min(min_vals)}, {max(min_vals)}]")
    print(f"    最大值范围: [{min(max_vals)}, {max(max_vals)}]")
    
    print(f"  rotation范围:")
    min_rots = [r[0] for r in rotation_ranges]
    max_rots = [r[1] for r in rotation_ranges]
    print(f"    最小值范围: [{min(min_rots)}, {max(min_rots)}]")
    print(f"    最大值范围: [{min(max_rots)}, {max(max_rots)}]")
    
    print(f"  效率: 最小={min(efficiencies):.4f}, 最大={max(efficiencies):.4f}, 平均={np.mean(efficiencies):.4f}")
    
    # 检查是否有异常的placement_order
    problematic_samples = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        seq_len = sample['valid_length']
        placement_order = sample['placement_order'].tolist()
        
        expected = set(range(seq_len))
        actual = set(placement_order)
        
        if expected != actual:
            problematic_samples.append(i)
    
    if problematic_samples:
        print(f"\n⚠️ 发现 {len(problematic_samples)} 个问题样本:")
        for i in problematic_samples[:5]:  # 只显示前5个
            sample = dataset[i]
            seq_len = sample['valid_length']
            placement_order = sample['placement_order'].tolist()
            print(f"  样本 {i}: seq_len={seq_len}, placement_order={placement_order}")
    else:
        print(f"\n✅ 前100个样本的placement_order都正常")


def main():
    print("🚀 开始调试模式...")
    
    # 1. 数据统计分析
    debug_data_statistics()
    
    # 2. 模型前向传播测试
    debug_model_forward()
    
    print("\n" + "=" * 60)
    print("🎉 调试完成! 请检查上述输出，确认数据和模型是否正常")
    print("=" * 60)


if __name__ == '__main__':
    main() 