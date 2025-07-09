import numpy as np
import torch
import sys
import os
import logging
from typing import List, Dict, Any, Tuple, Optional
import random
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.geometry_util import GeometryUtil
from utils.placement_worker import PlacementWorker
from src.reward.td_reward_calculator import TDRewardCalculator


logger = logging.getLogger(__name__)


class TDEnvironment:
    """时序差分环境 - 支持即时奖励计算和PlacementWorker集成"""
    
    def __init__(self, max_parts=60, reward_weights=(0.4, 0.3, 0.3)):
        """
        初始化环境
        
        Args:
            max_parts: 最大零件数量
            reward_weights: 三维奖励权重 (紧凑度, 贴合度, 材料利用率)
        """
        self.max_parts = max_parts
        
        # 初始化奖励计算器
        self.reward_calculator = TDRewardCalculator(
            w1=reward_weights[0], 
            w2=reward_weights[1], 
            w3=reward_weights[2]
        )
        
        # 环境状态
        self.reset()
        
        logger.info(f"TDEnvironment initialized with max_parts={max_parts}")

    def reset(self, instance: Optional[Dict] = None) -> np.ndarray:
        """重置环境"""
        if instance is not None:
            self.load_instance(instance)
        else:
            # 使用默认的测试实例
            self.load_default_instance()
            
        # 重置状态
        self.current_step = 0
        self.placement_sequence = []
        self.unplaced_parts = list(range(len(self.parts)))
        self.done = False
        self.total_reward = 0
        self.previous_layout_info = None
        self.step_rewards = []
        self.step_details = []
        
        return self._get_state()
    
    def load_instance(self, instance: Dict):
        """从实例数据加载bin和parts"""
        # 设置容器
        bin_width, bin_height = instance['bin']
        self.bin_bounds = {
            'x': 0, 'y': 0,
            'width': bin_width,
            'height': bin_height
        }
        
        # 创建容器多边形
        self.bin_polygon = [
            {'x': 0, 'y': 0},
            {'x': bin_width, 'y': 0},
            {'x': bin_width, 'y': bin_height},
            {'x': 0, 'y': bin_height},
            {'x': 0, 'y': 0}
        ]
        
        # 创建零件多边形 - 限制在最大数量内
        self.parts = []
        self.parts_info = []  # 存储原始的宽高信息
        max_parts_to_load = min(len(instance['parts']), self.max_parts)
        
        for i, part in enumerate(instance['parts'][:max_parts_to_load]):
            width, height = part
            # 创建自定义列表类
            CustomList = type('CustomList', (list,), {})
            polygon = CustomList([
                {'x': 0, 'y': 0},
                {'x': width, 'y': 0},
                {'x': width, 'y': height},
                {'x': 0, 'y': height},
                {'x': 0, 'y': 0}
            ])
            polygon.id = i
            self.parts.append(polygon)
            self.parts_info.append([width, height])
            
        self.num_parts = len(self.parts)
        
    def load_default_instance(self):
        """加载默认测试实例"""
        default_instance = {
            'bin': [100, 100],
            'parts': [[20, 30], [25, 25], [15, 40], [30, 20], [10, 50]]
        }
        self.load_instance(default_instance)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作 - 时序差分版本"""
        if self.done:
            return self._get_state(), 0, True, {}
            
        # 检查动作是否有效
        if action >= self.num_parts or action in self.placement_sequence:
            # 无效动作，给予负奖励
            reward = -1.0
            info = {
                'invalid_action': True,
                'step': self.current_step,
                'total_reward': self.total_reward
            }
            return self._get_state(), reward, self.done, info
        
        # 执行动作：将选中的零件添加到放置序列
        self.placement_sequence.append(action)
        if action in self.unplaced_parts:
            self.unplaced_parts.remove(action)
            
        self.current_step += 1
        
        # 计算即时奖励
        reward, reward_details = self._calculate_immediate_reward()
        self.total_reward += reward
        self.step_rewards.append(reward)
        self.step_details.append(reward_details)
        
        # 更新previous_layout_info用于下一步计算
        if 'layout_info' in reward_details:
            self.previous_layout_info = reward_details['layout_info']
        
        # 检查是否结束
        if len(self.placement_sequence) >= self.num_parts or len(self.unplaced_parts) == 0:
            self.done = True
            
        info = {
            'step': self.current_step,
            'placed_parts': len(self.placement_sequence),
            'total_reward': self.total_reward,
            'step_reward': reward,
            'reward_details': reward_details,
            'placement_sequence': self.placement_sequence.copy()
        }
        
        return self._get_state(), reward, self.done, info
    
    def _calculate_immediate_reward(self) -> Tuple[float, Dict]:
        """计算当前步骤的即时奖励"""
        try:
            reward, details = self.reward_calculator.calculate_step_reward(
                bin_polygon=self.bin_polygon,
                parts=self.parts,
                placement_sequence=self.placement_sequence,
                current_step=self.current_step - 1,  # 转换为0-based索引
                previous_layout_info=self.previous_layout_info
            )
            return reward, details
        except Exception as e:
            logger.warning(f"计算即时奖励时出错: {e}")
            return -1.0, {'error': str(e)}
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态 - 简化版本，只包含零件特征"""
        # 创建零件特征矩阵 [max_parts, 2]
        state = np.zeros((self.max_parts, 2), dtype=np.float32)
        
        # 填入零件信息（宽度，高度）
        for i, part_info in enumerate(self.parts_info):
            if i < self.max_parts:
                state[i, 0] = part_info[0] / 100.0  # 归一化宽度
                state[i, 1] = part_info[1] / 100.0  # 归一化高度
        
        return state
    
    def get_valid_actions(self) -> List[int]:
        """获取当前有效的动作"""
        return self.unplaced_parts.copy()
    
    def is_action_valid(self, action: int) -> bool:
        """检查动作是否有效"""
        return action < self.num_parts and action not in self.placement_sequence

    def get_placement_metrics(self) -> Dict:
        """获取当前排料指标"""
        if len(self.placement_sequence) == 0:
            return {
                'material_utilization': 0.0,
                'compactness': 0.0,
                'bbox_ratio': 0.0,
                'waste_area': 0.0,
                'final_score': 0.0
            }
        
        # 使用奖励计算器获取最终指标
        final_metrics = self.reward_calculator.calculate_final_reward(
            bin_polygon=self.bin_polygon,
            parts=self.parts,
            placement_sequence=self.placement_sequence
        )
        
        return final_metrics

    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Placement sequence: {self.placement_sequence}")
            print(f"Unplaced parts: {self.unplaced_parts}")
            print(f"Total reward: {self.total_reward:.3f}")
            if self.step_rewards:
                print(f"Last step reward: {self.step_rewards[-1]:.3f}")
            print("-" * 50)

    def get_episode_summary(self) -> Dict:
        """获取episode总结"""
        final_metrics = self.get_placement_metrics()
        
        return {
            'total_steps': self.current_step,
            'total_reward': self.total_reward,
            'average_step_reward': np.mean(self.step_rewards) if self.step_rewards else 0,
            'placement_sequence': self.placement_sequence,
            'final_metrics': final_metrics,
            'step_rewards': self.step_rewards,
            'num_parts': self.num_parts,
            'success': len(self.placement_sequence) == self.num_parts
        }

    def update_reward_weights(self, w1: float, w2: float, w3: float):
        """更新奖励权重"""
        self.reward_calculator.update_weights(w1, w2, w3)

    def get_reward_weights(self) -> Tuple[float, float, float]:
        """获取当前奖励权重"""
        return self.reward_calculator.get_weights()


class TDDataLoader:
    """时序差分数据加载器"""
    
    def __init__(self, data_file: str, test_data_file: str = None):
        self.data_file = data_file
        self.test_data_file = test_data_file
        self.instances = []
        self.test_instances = []
        self.load_data()
        if test_data_file:
            self.load_test_data()
    
    def load_data(self):
        """加载训练数据"""
        try:
            with open(self.data_file, 'r') as f:
                for line in f:
                    instance = json.loads(line)
                    # 保留bin、parts和placement_order信息
                    filtered_instance = {
                        'bin': instance['bin'],
                        'parts': instance['parts'],
                        'placement_order': instance.get('placement_order', [])
                    }
                    self.instances.append(filtered_instance)
            logger.info(f"加载了 {len(self.instances)} 个训练实例")
        except Exception as e:
            logger.error(f"加载训练数据时出错: {e}")
            self.instances = []
    
    def load_test_data(self):
        """加载测试数据"""
        try:
            with open(self.test_data_file, 'r') as f:
                for line in f:
                    instance = json.loads(line)
                    # 只保留bin和parts信息
                    filtered_instance = {
                        'bin': instance['bin'],
                        'parts': instance['parts']
                    }
                    self.test_instances.append(filtered_instance)
            logger.info(f"加载了 {len(self.test_instances)} 个测试实例")
        except Exception as e:
            logger.error(f"加载测试数据时出错: {e}")
            self.test_instances = []
    
    def get_random_instance(self) -> Dict:
        """获取随机训练实例"""
        if not self.instances:
            return None
        return random.choice(self.instances)
    
    def get_instance(self, index: int) -> Dict:
        """获取指定索引的训练实例"""
        if 0 <= index < len(self.instances):
            return self.instances[index]
        return None
    
    def get_test_instance(self, index: int) -> Dict:
        """获取指定索引的测试实例"""
        if 0 <= index < len(self.test_instances):
            return self.test_instances[index]
        return None
    
    def get_test_instances(self, num_instances: int = 100) -> List[Dict]:
        """获取指定数量的测试实例"""
        if not self.test_instances:
            return []
        return self.test_instances[:min(num_instances, len(self.test_instances))]
    
    def __len__(self):
        return len(self.instances) 