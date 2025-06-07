import numpy as np
import torch
import gym
from gym import spaces
from typing import List, Dict, Any, Tuple, Optional
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.geometry_util import GeometryUtil
from utils.placement_worker import PlacementWorker
import random
import json


class BinPackingEnvironment(gym.Env):
    """
    Bin Packing强化学习环境
    基于svgnest_v0.py中的placement方法
    """
    
    def __init__(self, max_parts=60, state_dim=256):
        super(BinPackingEnvironment, self).__init__()
        
        self.max_parts = max_parts
        self.state_dim = state_dim
        
        # Action space: 选择下一个要放置的零件
        self.action_space = spaces.Discrete(max_parts)
        
        # State space: 编码后的状态向量
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(state_dim,), dtype=np.float32
        )
        
        # 环境状态
        self.reset()
        
    def reset(self, instance: Optional[Dict] = None) -> np.ndarray:
        """重置环境"""
        if instance is not None:
            self.load_instance(instance)
        else:
            # 使用默认的测试实例
            self.load_default_instance()
            
        # 重置状态
        self.current_step = 0
        self.placed_parts = []
        self.unplaced_parts = list(range(len(self.parts)))
        self.current_placement_worker = None
        self.done = False
        self.total_reward = 0
        
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
        
        # 创建零件多边形
        self.parts = []
        self.parts_info = []  # 存储原始的宽高信息
        for i, part in enumerate(instance['parts']):
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
        """执行一步动作"""
        if self.done:
            return self._get_state(), 0, True, {}
            
        # 检查动作是否有效
        if action >= self.num_parts or action in self.placed_parts:
            # 无效动作，给予负奖励
            reward = -1.0
            return self._get_state(), reward, self.done, {'invalid_action': True}
        
        # 执行动作：将选中的零件添加到放置序列
        self.placed_parts.append(action)
        if action in self.unplaced_parts:
            self.unplaced_parts.remove(action)
            
        self.current_step += 1
        
        # 计算当前奖励
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # 检查是否结束
        if len(self.placed_parts) >= self.num_parts or len(self.unplaced_parts) == 0:
            self.done = True
            
        info = {
            'step': self.current_step,
            'placed_parts': len(self.placed_parts),
            'total_reward': self.total_reward,
            'efficiency': self._calculate_current_efficiency()
        }
        
        return self._get_state(), reward, self.done, info
    
    def _calculate_reward(self) -> float:
        """计算奖励函数"""
        if len(self.placed_parts) == 0:
            return 0.0
            
        # 使用PlacementWorker计算当前放置效率
        try:
            current_order = [self.parts[i] for i in self.placed_parts]
            rotations = [0] * len(current_order)  # 简化：不考虑旋转
            
            config = {
                'spacing': 0,
                'clipperScale': 10000000,
                'curveTolerance': 0
            }
            
            worker = PlacementWorker(
                bin_polygon=self.bin_polygon,
                paths=current_order,
                ids=self.placed_parts,
                rotations=rotations,
                config=config
            )
            
            placed, unplaced, bounding_box_area = worker.place_paths()
            
            # 计算零件总面积
            placed_parts_area = sum(abs(GeometryUtil.polygon_area(path)) for path in placed)
            
            # 计算真正的材料利用率：零件总面积 / 边界框面积
            if bounding_box_area > 0:
                material_utilization = placed_parts_area / bounding_box_area
            else:
                material_utilization = 0
                
            # 计算边界框相对于容器的占用比（越小越好，用于鼓励紧密排列）
            bin_area = abs(GeometryUtil.polygon_area(self.bin_polygon))
            if bin_area > 0:
                bbox_ratio = bounding_box_area / bin_area  # 边界框占容器的比例
            else:
                bbox_ratio = 1.0
                
            # 奖励设计
            reward = 0.0
            
            # 1. 基础效率奖励
            reward += material_utilization * 10.0
            
            # 2. 增量奖励：相比上一步的改进
            if hasattr(self, 'prev_efficiency'):
                improvement = material_utilization - self.prev_efficiency
                if improvement > 0:
                    reward += improvement * 5.0  # 放大改进奖励
            
            self.prev_efficiency = material_utilization
            
            # 3. 完成奖励 - 分段阶梯式 + 线性奖励
            if self.done:
                # 分段阶梯式奖励（保持原有逻辑）
                if material_utilization > 0.9:  # 高效率完成
                    reward += 50.0
                elif material_utilization > 0.8:  # 中等效率完成
                    reward += 30.0
                elif material_utilization > 0.7:  # 低效率完成
                    reward += 10.0
                else:  # 低效率完成
                    reward += 1.0
                
                # 在分段奖励基础上，再增加线性奖励
                linear_reward = material_utilization * 50.0  # 利用率线性奖励 (0-50)
                reward += linear_reward
            
            return reward
            
        except Exception as e:
            print(f"计算奖励时出错: {e}")
            return -5.0  # 出错时给予负奖励
    
    def _calculate_current_efficiency(self) -> float:
        """计算当前材料利用率"""
        if len(self.placed_parts) == 0:
            return 0.0
            
        try:
            current_order = [self.parts[i] for i in self.placed_parts]
            rotations = [0] * len(current_order)
            
            config = {
                'spacing': 0,
                'clipperScale': 10000000,
                'curveTolerance': 0
            }
            
            worker = PlacementWorker(
                bin_polygon=self.bin_polygon,
                paths=current_order,
                ids=self.placed_parts,
                rotations=rotations,
                config=config
            )
            
            placed, unplaced, bounding_box_area = worker.place_paths()
            
            # 计算零件总面积
            placed_parts_area = sum(abs(GeometryUtil.polygon_area(path)) for path in placed)
            
            # 计算真正的材料利用率：零件总面积 / 边界框面积
            if bounding_box_area > 0:
                return placed_parts_area / bounding_box_area
            else:
                return 0.0
                
        except Exception as e:
            print(f"计算效率时出错: {e}")
            return 0.0
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态编码"""
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # 基础信息编码
        state[0] = len(self.placed_parts) / self.max_parts  # 已放置比例
        state[1] = len(self.unplaced_parts) / self.max_parts  # 未放置比例
        state[2] = self.current_step / self.max_parts  # 步数比例
        
        # 容器信息编码
        state[3] = self.bin_bounds['width'] / 200.0  # 归一化容器宽度
        state[4] = self.bin_bounds['height'] / 200.0  # 归一化容器高度
        
        # 当前材料利用率和紧密度信息
        state[5] = self._calculate_current_efficiency()  # 材料利用率
        
        # 获取详细指标
        if len(self.placed_parts) > 0:
            metrics = self.get_placement_metrics()
            state[6] = metrics['bbox_ratio']  # 边界框占容器比例
            state[7] = metrics['compactness_score']  # 紧密度评分
        else:
            state[6] = 0.0
            state[7] = 0.0
            
        # 历史改进信息
        if hasattr(self, 'prev_material_utilization'):
            state[8] = self.prev_material_utilization  # 上一步的材料利用率
        else:
            state[8] = 0.0
            
        if hasattr(self, 'prev_bbox_ratio'):
            state[9] = self.prev_bbox_ratio  # 上一步的边界框比例
        else:
            state[9] = 0.0
        
        # 零件信息编码
        start_idx = 10  # 前面使用了索引0-9
        max_parts_to_encode = min(len(self.parts_info), self.max_parts)  # 编码所有零件
        
        # 计算可用空间：确保不超出state_dim
        available_space = self.state_dim - start_idx
        parts_can_encode = min(max_parts_to_encode, available_space // 3)  # 每个零件3个维度
        
        for i in range(parts_can_encode):
            if i < len(self.parts_info):
                part_info = self.parts_info[i]
                base_idx = start_idx + i * 3
                
                # 确保不越界
                if base_idx + 2 < self.state_dim:
                    state[base_idx] = part_info[0] / 100.0      # 归一化宽度
                    state[base_idx + 1] = part_info[1] / 100.0  # 归一化高度
                    state[base_idx + 2] = 1.0 if i in self.placed_parts else 0.0  # 是否已放置
        
        # # 添加随机噪声以增加探索
        # noise_level = 0.01
        # noise = np.random.normal(0, noise_level, state.shape)
        # state = state + noise
        
        return state.astype(np.float32)
    
    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Placed parts: {self.placed_parts}")
            print(f"Unplaced parts: {self.unplaced_parts}")
            print(f"Current efficiency: {self._calculate_current_efficiency():.3f}")
            print(f"Total reward: {self.total_reward:.3f}")
            print("-" * 50)
    
    def get_valid_actions(self) -> List[int]:
        """获取当前有效的动作"""
        return self.unplaced_parts.copy()
    
    def is_action_valid(self, action: int) -> bool:
        """检查动作是否有效"""
        return action < self.num_parts and action not in self.placed_parts

    def get_placement_metrics(self) -> Dict:
        """获取详细的排料指标"""
        if len(self.placed_parts) == 0:
            return {
                'material_utilization': 0.0,
                'bbox_ratio': 0.0,
                'placed_parts_area': 0.0,
                'bounding_box_area': 0.0,
                'compactness_score': 0.0
            }
            
        try:
            current_order = [self.parts[i] for i in self.placed_parts]
            rotations = [0] * len(current_order)
            
            config = {
                'spacing': 0,
                'clipperScale': 10000000,
                'curveTolerance': 0
            }
            
            worker = PlacementWorker(
                bin_polygon=self.bin_polygon,
                paths=current_order,
                ids=self.placed_parts,
                rotations=rotations,
                config=config
            )
            
            placed, unplaced, bounding_box_area = worker.place_paths()
            
            # 计算零件总面积
            placed_parts_area = sum(abs(GeometryUtil.polygon_area(path)) for path in placed)
            
            # 计算材料利用率
            material_utilization = placed_parts_area / bounding_box_area if bounding_box_area > 0 else 0
            
            # 计算边界框占容器比例
            bin_area = abs(GeometryUtil.polygon_area(self.bin_polygon))
            bbox_ratio = bounding_box_area / bin_area if bin_area > 0 else 1.0
            
            # 计算紧密度评分（综合指标）
            compactness_score = material_utilization * (1.0 - bbox_ratio)
            
            return {
                'material_utilization': material_utilization,
                'bbox_ratio': bbox_ratio,
                'placed_parts_area': placed_parts_area,
                'bounding_box_area': bounding_box_area,
                'compactness_score': compactness_score
            }
            
        except Exception as e:
            print(f"计算排料指标时出错: {e}")
            return {
                'material_utilization': 0.0,
                'bbox_ratio': 0.0,
                'placed_parts_area': 0.0,
                'bounding_box_area': 0.0,
                'compactness_score': 0.0
            }


class BinPackingDataLoader:
    """数据加载器，用于从JSONL文件加载实例"""
    
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
                    self.instances.append(instance)
            print(f"加载了 {len(self.instances)} 个训练实例")
        except Exception as e:
            print(f"加载训练数据时出错: {e}")
            self.instances = []
    
    def load_test_data(self):
        """加载测试数据"""
        try:
            with open(self.test_data_file, 'r') as f:
                for line in f:
                    instance = json.loads(line)
                    self.test_instances.append(instance)
            print(f"加载了 {len(self.test_instances)} 个测试实例")
        except Exception as e:
            print(f"加载测试数据时出错: {e}")
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
        """获取指定数量的测试实例（从前面开始）"""
        if not self.test_instances:
            return []
        return self.test_instances[:min(num_instances, len(self.test_instances))]
    
    def __len__(self):
        return len(self.instances) 