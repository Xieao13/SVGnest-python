import json
import os
import random


def split_simple_dataset(input_file, train_output_file, test_output_file, split_ratio=0.8):
    """将简单的JSONL数据集文件分割成训练集和测试集"""
    
    print(f"📂 正在加载数据文件: {input_file}")
    
    data = []
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line)
                    # 简单验证
                    if 'bin' in item and 'parts' in item and len(item['parts']) > 0:
                        data.append(item)
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行JSON解析失败: {e}")
                    continue
    
    if not data:
        print(f"⚠️ 警告: 没有有效数据可以分割")
        return
    
    print(f"  成功加载 {len(data)} 条数据")
    print(f"\n🔄 开始分割数据集...")
    
    # 使用random.shuffle进行分割（固定随机种子）
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # 确保输出目录存在
    for output_file in [train_output_file, test_output_file]:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # 将分割后的数据写入新的JSONL文件
    with open(train_output_file, 'w') as f:
        for entry in train_data:
            json.dump(entry, f)
            f.write('\n')
    
    with open(test_output_file, 'w') as f:
        for entry in test_data:
            json.dump(entry, f)
            f.write('\n')
    
    print(f"✅ 数据集分割完成:")
    print(f"  训练集: {len(train_data)} 条数据 -> {train_output_file}")
    print(f"  测试集: {len(test_data)} 条数据 -> {test_output_file}")
    print(f"  分割比例: {len(train_data)/(len(train_data)+len(test_data))*100:.1f}% / {len(test_data)/(len(train_data)+len(test_data))*100:.1f}%")


if __name__ == "__main__":
    folder_path = './data/simple_test'
    input_dataset_path = os.path.join(folder_path, 'instances.jsonl')
    train_output_path = os.path.join(folder_path, 'train.jsonl')
    test_output_path = os.path.join(folder_path, 'test.jsonl')
    
    # 检查输入文件是否存在
    if not os.path.exists(input_dataset_path):
        print(f"❌ 错误: 输入数据文件不存在 {input_dataset_path}")
        print(f"请先运行 create_simple_dataset.py 生成数据")
    else:
        print("🚀 开始数据分割...")
        split_simple_dataset(input_dataset_path, train_output_path, test_output_path)

