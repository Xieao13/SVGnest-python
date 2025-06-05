# import json

# # 假设这个变量已经定义
# corrupted_file_path = "/Users/xieao/Documents/new_svgnest/src/results/predictions_20250508_150412.jsonl"
# fixed_file_path = "/Users/xieao/Documents/new_svgnest/src/results/predictions_20250508_150412.jsonl"

# # 读取整个文件内容
# with open(corrupted_file_path, 'r') as f:
#     content = f.read()

# # 使用 JSON 解析器将内容解析为对象列表
# predictions = json.loads(f"[{content}]")

# # 将每个对象逐行写入新的 JSON Lines 文件
# with open(fixed_file_path, 'w') as f:
#     for prediction in predictions:
#         json.dump(prediction, f)
#         f.write('\n')  # 添加换行符

# print(f"Fixed JSON Lines file saved to {fixed_file_path}")
# import json

# # 假设这些变量已经定义
# corrupted_file_path = "/Users/xieao/Documents/new_svgnest/src/results/predictions_20250508_150412.jsonl"
# fixed_file_path = "/Users/xieao/Documents/new_svgnest/src/results/predictions_20250508_150412.jsonl"

# # 读取文件内容
# with open(corrupted_file_path, 'r') as f:
#     lines = f.readlines()

# # 在每个 } 符号后面加一个换行符
# fixed_content = []
# for line in lines:
#     fixed_line = line.replace('}', '}\n')
#     fixed_content.append(fixed_line.strip())

# # 将修复后的内容写入新的 JSON Lines 文件
# with open(fixed_file_path, 'w') as f:
#     f.write(',\n'.join(fixed_content))
# print(f"Fixed JSON Lines file saved to {fixed_file_path}")

# 读取文件内容
import json

def calculate_average_efficiency(file_path):
    total_efficiency = 0
    count = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            if count > 1000:
                break
            data = json.loads(line)
            efficiency = data.get("efficiency")
            if efficiency is not None:
                total_efficiency += efficiency
                count += 1
    
    if count > 0:
        return total_efficiency / count
    else:
        return 0

model_result_path = "/Users/xieao/Documents/new_svgnest/output/placement_mymodel_0514.jsonl"
ga_result_path = "/Users/xieao/Documents/new_svgnest/output/placement_ga_0514.jsonl"
random_result_path = "/Users/xieao/Documents/new_svgnest/output/placement_random_0514.jsonl"
average_efficiency_model = calculate_average_efficiency(model_result_path)
average_efficiency_ga = calculate_average_efficiency(ga_result_path)
average_efficiency_random = calculate_average_efficiency(random_result_path)
print(f"Model Average Efficiency: {average_efficiency_model}")
print(f"GA Average Efficiency: {average_efficiency_ga}")
print(f"Random Average Efficiency: {average_efficiency_random}")

import json

# def write_first_n_lines_to_jsonl(input_file_path, output_file_path, n=50000):
#     with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
#         for i, line in enumerate(infile):
#             if i >= n:
#                 break
#             outfile.write(line)

# n = 50000  # 你可以在这里灵活设置要读取的数据量

# jsonl_file = "output/instances.jsonl"
# output_file = f"output/instances_{n}.jsonl"


# write_first_n_lines_to_jsonl(jsonl_file, output_file, n)

# print(f"前{n}个数据已写入到 {output_file}")


