import json
import os
import random
from sklearn.model_selection import train_test_split

def validate_data_item(item, line_num):
    """验证单个数据项的有效性"""
    try:
        # 检查必要字段
        required_fields = ['bin', 'parts', 'placement_order', 'efficiency']
        for field in required_fields:
            if field not in item:
                print(f"第{line_num}行缺少字段: {field}")
                return False
        
        # 检查长度一致性
        parts_len = len(item['parts'])
        placement_order_len = len(item['placement_order'])
        
        # 如果有rotation字段，也检查其长度
        if 'rotation' in item:
            rotation_len = len(item['rotation'])
            if parts_len != placement_order_len or parts_len != rotation_len:
                print(f"第{line_num}行长度不一致: parts({parts_len}) vs placement_order({placement_order_len}) vs rotation({rotation_len})")
                return False
        else:
            if parts_len != placement_order_len:
                print(f"第{line_num}行长度不一致: parts({parts_len}) vs placement_order({placement_order_len})")
                return False
        
        # 检查placement_order是否是有效排列
        expected_indices = set(range(parts_len))
        actual_indices = set(item['placement_order'])
        
        if expected_indices != actual_indices:
            missing = expected_indices - actual_indices
            extra = actual_indices - expected_indices
            print(f"第{line_num}行placement_order不是有效排列")
            if missing:
                print(f"  缺失索引: {sorted(missing)}")
            if extra:
                print(f"  多余索引: {sorted(extra)}")
            return False
        
        # 检查索引范围
        max_index = max(item['placement_order'])
        min_index = min(item['placement_order'])
        if max_index >= parts_len or min_index < 0:
            print(f"第{line_num}行placement_order索引超出范围: [{min_index}, {max_index}], 应该在[0, {parts_len-1}]")
            return False
        
        # 检查是否有重复
        if len(set(item['placement_order'])) != len(item['placement_order']):
            print(f"第{line_num}行placement_order有重复值")
            return False
        
        # 检查parts是否为空或包含无效值
        if not item['parts'] or any(not isinstance(part, list) or len(part) != 2 for part in item['parts']):
            print(f"第{line_num}行parts格式无效")
            return False
        
        # 检查bin格式
        if not isinstance(item['bin'], list) or len(item['bin']) != 2:
            print(f"第{line_num}行bin格式无效")
            return False
        
        return True
        
    except Exception as e:
        print(f"第{line_num}行数据验证失败: {e}")
        return False

def load_and_filter_data(input_file):
    """加载并过滤数据"""
    data = []
    total_loaded = 0
    valid_count = 0
    
    print(f"📂 正在加载和验证数据文件: {input_file}")
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line)
                    total_loaded += 1
                    
                    # 验证数据完整性
                    if validate_data_item(item, line_num):
                        data.append(item)
                        valid_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行JSON解析失败: {e}")
                    continue
    
    print(f"📊 数据加载统计:")
    print(f"  总共读取: {total_loaded} 条数据")
    print(f"  有效数据: {valid_count} 条")
    print(f"  过滤掉: {total_loaded - valid_count} 条无效数据")
    print(f"  有效率: {valid_count/total_loaded*100:.2f}%" if total_loaded > 0 else "  有效率: 0%")
    
    return data

def split_dataset(input_file, train_output_file, test_output_file, split_ratio=0.8):
    """将JSONL数据集文件分割成训练集和测试集（先过滤无效数据）"""
    
    # 加载并过滤数据
    data = load_and_filter_data(input_file)

    if not data:
        print(f"⚠️ 警告: 没有有效数据可以分割")
        return

    print(f"\n🔄 开始分割数据集...")
    
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

    print(f"✅ 数据集分割完成:")
    print(f"  训练集: {len(train_data)} 条数据 -> {train_output_file}")
    print(f"  测试集: {len(test_data)} 条数据 -> {test_output_file}")
    print(f"  分割比例: {len(train_data)/(len(train_data)+len(test_data))*100:.1f}% / {len(test_data)/(len(train_data)+len(test_data))*100:.1f}%")

if __name__ == "__main__":
    folder_path = './data/placement-0529-ga-20epoch-norotation'
    input_dataset_path = os.path.join(folder_path, 'raw_data.jsonl')
    train_output_path = os.path.join(folder_path, 'train.jsonl')
    test_output_path = os.path.join(folder_path, 'test.jsonl')

    # 检查输入文件是否存在
    if not os.path.exists(input_dataset_path):
        print(f"❌ 错误: 输入数据文件不存在 {input_dataset_path}")
        print(f"请确保原始数据文件存在")
    else:
        print("🚀 开始数据验证和分割...")
        split_dataset(input_dataset_path, train_output_path, test_output_path)
