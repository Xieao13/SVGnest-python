#!/bin/bash
# 使用遗传算法测试简单数据集

echo "🔧 使用遗传算法测试简单数据集"
echo ""

# 创建输出目录
mkdir -p ./output/simple_test_ga

# 使用 svgnest_ga.py 处理简单数据集的样本
python3 svgnest_ga.py \
    --input_file "./data/simple_test/sample_5.jsonl" \
    --output_file "./output/simple_test_ga/results.jsonl" 2>&1 | head -200

echo ""
echo "测试完成！"

