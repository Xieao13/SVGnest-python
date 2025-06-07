#!/usr/bin/env python3
"""
测试数据加载和过滤功能
"""

import sys
import os
import logging

# 添加src目录到路径
sys.path.append('src')

from dataset.dataset import PlacementOnlyDataset, PlacementOnly_collate_fn
from torch.utils.data import DataLoader

# 设置日志级别
logging.basicConfig(level=logging.INFO)

def test_data_loading():
    """测试数据加载和过滤"""
    print("=" * 80)
    print("🔍 测试数据加载和过滤功能")
    print("=" * 80)
    
    # 测试加载训练数据
    train_file = "./data/placement-0529-ga-20epoch-norotation/train.jsonl"
    
    if not os.path.exists(train_file):
        print(f"❌ 数据文件不存在: {train_file}")
        return False
    
    try:
        print(f"📂 加载数据文件: {train_file}")
        dataset = PlacementOnlyDataset(train_file)
        
        print(f"✅ 数据加载完成")
        print(f"  有效数据样本数: {len(dataset)}")
        
        # 测试前几个样本
        print(f"\n🔍 检查前5个样本:")
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            parts_len = len(sample['parts'])
            placement_len = len(sample['placement_order'])
            
            print(f"  样本 {i}: parts长度={parts_len}, placement_order长度={placement_len}")
            
            # 验证是否是有效排列
            expected = set(range(parts_len))
            actual = set(sample['placement_order'].tolist())
            if expected == actual:
                print(f"    ✅ 有效排列")
            else:
                print(f"    ❌ 无效排列: 期望{expected}, 实际{actual}")
                return False
        
        # 测试DataLoader
        print(f"\n🔍 测试DataLoader:")
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False, 
            collate_fn=PlacementOnly_collate_fn
        )
        
        # 测试第一个batch
        for batch_idx, batch in enumerate(dataloader):
            print(f"  Batch {batch_idx}:")
            print(f"    parts shape: {batch['parts'].shape}")
            print(f"    placement_orders shape: {batch['placement_orders'].shape}")
            print(f"    valid_lengths: {batch['valid_lengths']}")
            
            # 验证padding
            for i in range(batch['placement_orders'].shape[0]):
                valid_len = batch['valid_lengths'][i].item()
                placement_order = batch['placement_orders'][i, :valid_len]
                
                # 检查是否是有效排列
                expected = set(range(valid_len))
                actual = set(placement_order.tolist())
                if expected == actual:
                    print(f"    样本 {i}: ✅ 有效排列")
                else:
                    print(f"    样本 {i}: ❌ 无效排列")
                    return False
            
            # 只测试第一个batch
            break
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始数据加载和过滤测试")
    
    if test_data_loading():
        print("\n🎉 数据加载测试通过！")
        print("现在可以安全地进行训练了")
        return True
    else:
        print("\n❌ 数据加载测试失败")
        return False

if __name__ == "__main__":
    main() 