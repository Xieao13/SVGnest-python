import json
import os
import random
from sklearn.model_selection import train_test_split

def split_dataset(input_file, train_output_file, test_output_file, split_ratio=0.8):
    """将JSONL数据集文件分割成训练集和测试集"""
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    if not data:
        print(f"Warning: Input file {input_file} is empty or could not be read.")
        return

    # 使用train_test_split进行分割，确保固定种子以便结果可复现
    train_data, test_data = train_test_split(data, train_size=split_ratio, random_state=42)

    # 确保输出目录存在
    output_dir = os.path.dirname(train_output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_dir = os.path.dirname(test_output_file)
    if not os.path.exists(output_dir):
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

    print(f"数据集分割完成:\n  训练集: {len(train_data)} 条数据 -> {train_output_file}\n  测试集: {len(test_data)} 条数据 -> {test_output_file}")

if __name__ == "__main__":
    folder_path = './data/placement-0529-ga-20epoch-norotation'
    input_dataset_path = os.path.join(folder_path, 'raw_data.jsonl')
    train_output_path = os.path.join(folder_path, 'train.jsonl')
    test_output_path = os.path.join(folder_path, 'test.jsonl')

    # 检查输入文件是否存在
    if not os.path.exists(input_dataset_path):
        print(f"Error: Input dataset file not found at {input_dataset_path}")
    else:
        split_dataset(input_dataset_path, train_output_path, test_output_path)
