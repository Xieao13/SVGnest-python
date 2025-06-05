import random
import os
import json


def generate_instance(bin_width, bin_height, num_items):
    items = []
    profit = 0
    for _ in range(num_items):
        item_width = random.randint(10, bin_width//4)
        item_height = random.randint(10, bin_height//4)
        item_profit = item_width * item_height
        profit += item_profit
        if profit > (bin_width * bin_height)*0.6:
            break
        items.append([item_width, item_height])
    
    return {
        "bin": [bin_width, bin_height],
        "parts": items
    }


def main():
    num_files = 100000
    min_items = 20  # Minimum number of items
    max_items = 100  # Maximum number of items
    max_width = 1000
    max_height = 1000
    output_file = "output/instances.jsonl"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 打开文件准备写入
    with open(output_file, 'w') as f:
        for i in range(num_files):
            num_items = random.randint(min_items, max_items)
            bin_width = random.randint(400, max_width)
            bin_height = random.randint(400, max_height)
            
            # 生成一个实例
            instance = generate_instance(bin_width, bin_height, num_items)
            
            # 将实例写入JSONL文件
            f.write(json.dumps(instance) + '\n')


if __name__ == "__main__":
    main()
