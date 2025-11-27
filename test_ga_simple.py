#!/usr/bin/env python3
"""
使用遗传算法测试简单数据集的排布效果
"""
import json
import os
import sys
from svgnest_ga import SvgNest
from tqdm import tqdm

def test_simple_dataset(input_file, output_file, num_instances=10):
    """测试简单数据集的前N个实例"""
    
    print(f"🔧 使用遗传算法测试简单数据集")
    print(f"  输入文件: {input_file}")
    print(f"  输出文件: {output_file}")
    print(f"  测试实例数: {num_instances}")
    print()
    
    # 创建SvgNest实例
    nester = SvgNest()
    
    # 配置参数（针对简单数据集优化）
    config = {
        'spacing': 0,
        'rotations': 1,  # 不旋转，简化测试
        'populationSize': 20,  # 减少种群规模以加快速度
        'mutationRate': 20,
        'useHoles': True,
        'exploreConcave': False
    }
    print(f"配置参数: {config}")
    nester.set_config(config)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    efficiencies = []
    
    # 读取并处理实例
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()[:num_instances]  # 只处理前N个
        
        for i, line in enumerate(tqdm(lines, total=len(lines), desc="处理实例")):
            try:
                instance = json.loads(line)
                print(f"\n{'='*60}")
                print(f"实例 {i+1}/{len(lines)}")
                print(f"  Bin大小: {instance['bin']}")
                print(f"  零件数量: {len(instance['parts'])}")
                
                # 加载实例数据
                if not nester.load_from_jsonl(instance):
                    print("  ❌ 加载实例数据失败")
                    continue
                
                nester.set_config(config)
                
                # 开始布局计算
                print("  开始布局计算...")
                result = nester.start(is_svg=False)
                if not result:
                    print("  ❌ 布局计算失败")
                    continue
                
                # 获取最佳结果
                if nester.best:
                    efficiency = nester.best.get('efficiency', 0)
                    efficiencies.append(efficiency)
                    
                    # 提取零件顺序
                    placement_order = [p.id for p in nester.best['paths']]
                    
                    result_data = {
                        'bin': instance['bin'],
                        'parts': instance['parts'],
                        'placement_order': placement_order,
                        'efficiency': efficiency
                    }
                    results.append(result_data)
                    
                    print(f"  ✅ 完成 - 材料利用率: {efficiency:.2%}")
                    print(f"  放置顺序: {placement_order}")
                else:
                    print("  ❌ 未找到有效结果")
                    
            except Exception as e:
                print(f"  ❌ 处理实例时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 保存结果
    if results:
        with open(output_file, 'w') as f_out:
            for result in results:
                f_out.write(json.dumps(result) + '\n')
        
        # 统计信息
        print(f"\n{'='*60}")
        print(f"📊 测试结果统计:")
        print(f"  成功处理: {len(results)}/{num_instances} 个实例")
        if efficiencies:
            avg_efficiency = sum(efficiencies) / len(efficiencies)
            min_efficiency = min(efficiencies)
            max_efficiency = max(efficiencies)
            print(f"  平均材料利用率: {avg_efficiency:.2%}")
            print(f"  最低材料利用率: {min_efficiency:.2%}")
            print(f"  最高材料利用率: {max_efficiency:.2%}")
        print(f"  结果已保存到: {output_file}")
    else:
        print("\n❌ 没有成功处理任何实例")

if __name__ == '__main__':
    input_file = "./data/simple_test/instances.jsonl"
    output_file = "./output/simple_test_ga_results.jsonl"
    
    # 测试前10个实例
    test_simple_dataset(input_file, output_file, num_instances=10)

