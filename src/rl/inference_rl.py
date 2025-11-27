import numpy as np
import torch
import os
import sys
import json
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.rl.environment import BinPackingEnvironment, BinPackingDataLoader
from src.rl.dqn_agent import DQNAgent


class RLInference:
    """强化学习模型推理器"""
    
    def __init__(self, model_path: str, max_parts: int = 60, state_dim: int = 128):
        self.max_parts = max_parts
        self.state_dim = state_dim
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建环境
        self.env = BinPackingEnvironment(
            max_parts=max_parts,
            state_dim=state_dim
        )
        
        # 创建智能体
        self.agent = DQNAgent(
            state_dim=state_dim,
            action_dim=max_parts,
            device=self.device
        )
        
        # 加载模型
        self.load_model(model_path)
        self.agent.set_eval_mode()
        
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        print(f"加载模型: {model_path}")
        self.agent.load_model(model_path)
        
    def predict_placement_order(self, instance: Dict) -> Tuple[List[int], float]:
        """
        预测放置顺序
        
        Args:
            instance: 包含bin和parts信息的字典
            
        Returns:
            placement_order: 预测的放置顺序
            efficiency: 预测的放置效率
        """
        # 重置环境
        state = self.env.reset(instance)
        placement_order = []
        steps = 0
        max_steps = len(self.env.parts) + 10  # 防止无限循环
        
        while not self.env.done and steps < max_steps:
            # 获取有效动作
            valid_actions = self.env.get_valid_actions()
            if not valid_actions:
                break
                
            # 使用贪婪策略选择动作（epsilon=0）
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.agent.policy_net(state_tensor)
                
                # 只考虑有效动作
                masked_q_values = q_values.clone()
                invalid_actions = [a for a in range(self.max_parts) if a not in valid_actions]
                if invalid_actions:
                    masked_q_values[0, invalid_actions] = float('-inf')
                
                action = masked_q_values.argmax().item()
            
            # 执行动作
            state, reward, done, info = self.env.step(action)
            placement_order.append(action)
            steps += 1
        
        # 计算最终效率
        final_efficiency = self.env._calculate_current_efficiency()
        
        return placement_order, final_efficiency
    
    def predict_batch(self, instances: List[Dict]) -> List[Dict]:
        """
        批量预测
        
        Args:
            instances: 实例列表
            
        Returns:
            results: 预测结果列表
        """
        results = []
        
        for i, instance in enumerate(tqdm(instances, desc="批量预测")):
            try:
                placement_order, efficiency = self.predict_placement_order(instance)
                
                result = {
                    'instance_id': i,
                    'bin': instance['bin'],
                    'parts': instance['parts'],
                    'predicted_placement': placement_order,
                    'predicted_rotation': [0] * len(placement_order),  # 简化：不预测旋转
                    'predicted_efficiency': efficiency
                }
                results.append(result)
                
            except Exception as e:
                print(f"预测实例 {i} 时出错: {e}")
                # 添加默认结果
                result = {
                    'instance_id': i,
                    'bin': instance['bin'],
                    'parts': instance['parts'],
                    'predicted_placement': list(range(len(instance['parts']))),
                    'predicted_rotation': [0] * len(instance['parts']),
                    'predicted_efficiency': 0.0
                }
                results.append(result)
        
        return results
    
    def evaluate_on_dataset(self, data_file: str, output_file: str = None) -> Dict:
        """
        在数据集上评估模型性能
        
        Args:
            data_file: 数据文件路径
            output_file: 输出文件路径（可选）
            
        Returns:
            evaluation_stats: 评估统计信息
        """
        # 加载数据
        data_loader = BinPackingDataLoader(data_file)
        print(f"加载了 {len(data_loader)} 个实例用于评估")
        
        # 批量预测
        results = self.predict_batch(data_loader.instances)
        
        # 计算统计信息
        efficiencies = [r['predicted_efficiency'] for r in results]
        
        stats = {
            'num_instances': len(results),
            'avg_efficiency': np.mean(efficiencies),
            'std_efficiency': np.std(efficiencies),
            'min_efficiency': np.min(efficiencies),
            'max_efficiency': np.max(efficiencies),
            'median_efficiency': np.median(efficiencies)
        }
        
        print(f"评估结果:")
        print(f"  实例数量: {stats['num_instances']}")
        print(f"  平均效率: {stats['avg_efficiency']:.3f}")
        print(f"  效率标准差: {stats['std_efficiency']:.3f}")
        print(f"  最小效率: {stats['min_efficiency']:.3f}")
        print(f"  最大效率: {stats['max_efficiency']:.3f}")
        print(f"  中位数效率: {stats['median_efficiency']:.3f}")
        
        # 保存结果
        if output_file:
            print(f"保存预测结果到: {output_file}")
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="RL Model Inference for Bin Packing")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained RL model")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the test data file")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save prediction results")
    parser.add_argument("--max_parts", type=int, default=60,
                        help="Maximum number of parts")
    parser.add_argument("--state_dim", type=int, default=128,
                        help="State dimension")
    
    args = parser.parse_args()
    
    # 创建推理器
    print("初始化RL推理器...")
    inference = RLInference(
        model_path=args.model_path,
        max_parts=args.max_parts,
        state_dim=args.state_dim
    )
    
    # 执行评估
    print("开始评估...")
    stats = inference.evaluate_on_dataset(
        data_file=args.data_file,
        output_file=args.output_file
    )
    
    print("评估完成！")


if __name__ == '__main__':
    main() 