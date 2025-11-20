import random
import os
import json


def generate_simple_instance(bin_width, bin_height, num_items):
    """生成简单的实例，确保零件能放入bin"""
    items = []
    total_area = 0
    max_area = bin_width * bin_height * 0.7  # 最多使用70%的面积
    
    for i in range(num_items):
        # 生成较小的零件，确保能放入
        item_width = random.randint(10, min(50, bin_width // 3))
        item_height = random.randint(10, min(50, bin_height // 3))
        
        item_area = item_width * item_height
        if total_area + item_area > max_area:
            break
        
        items.append([item_width, item_height])
        total_area += item_area
    
    return {
        "bin": [bin_width, bin_height],
        "parts": items
    }


def main():
    # 简化参数：更少的实例、更小的尺寸、更少的零件
    num_instances = 1000  # 减少到1000个实例用于快速测试
    min_items = 5   # 最少5个零件
    max_items = 15  # 最多15个零件（原来20-100）
    min_bin_size = 100
    max_bin_size = 300  # 更小的bin（原来400-1000）
    
    output_dir = "./data/simple_test"
    output_file = os.path.join(output_dir, "instances.jsonl")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔧 生成简化数据集...")
    print(f"  实例数量: {num_instances}")
    print(f"  零件数量: {min_items}-{max_items}")
    print(f"  Bin大小: {min_bin_size}-{max_bin_size}")
    print(f"  输出文件: {output_file}")
    
    # 打开文件准备写入
    with open(output_file, 'w') as f:
        for i in range(num_instances):
            num_items = random.randint(min_items, max_items)
            bin_width = random.randint(min_bin_size, max_bin_size)
            bin_height = random.randint(min_bin_size, max_bin_size)
            
            # 生成一个实例
            instance = generate_simple_instance(bin_width, bin_height, num_items)
            
            # 确保至少有一些零件
            if len(instance['parts']) < min_items:
                continue
            
            # 将实例写入JSONL文件
            f.write(json.dumps(instance) + '\n')
            
            if (i + 1) % 100 == 0:
                print(f"  已生成 {i + 1}/{num_instances} 个实例...")
    
    print(f"✅ 数据集生成完成: {output_file}")
    
    # 统计信息
    with open(output_file, 'r') as f:
        instances = [json.loads(line) for line in f]
    
    num_parts_list = [len(inst['parts']) for inst in instances]
    bin_sizes = [(inst['bin'][0], inst['bin'][1]) for inst in instances]
    
    print(f"\n📊 数据集统计:")
    print(f"  总实例数: {len(instances)}")
    print(f"  零件数量: 平均={sum(num_parts_list)/len(num_parts_list):.1f}, "
          f"最小={min(num_parts_list)}, 最大={max(num_parts_list)}")
    print(f"  Bin大小: 平均={sum(w+h for w,h in bin_sizes)/len(bin_sizes)/2:.0f}, "
          f"范围=[{min_bin_size}, {max_bin_size}]")


if __name__ == "__main__":
    main()

