import numpy as np
import torch
import sys
import os
from typing import List, Dict, Tuple, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.geometry_util import GeometryUtil
from utils.placement_worker import PlacementWorker


class TDRewardCalculator:
    """时序差分奖励计算器 - 实现三维奖励函数"""
    
    def __init__(self, w1=0.4, w2=0.3, w3=0.3, spacing=0):
        """
        初始化奖励计算器
        
        Args:
            w1: 紧凑度奖励权重
            w2: 贴合度奖励权重  
            w3: 材料利用率奖励权重
            spacing: 零件间距
        """
        # 权重归一化
        total_weight = w1 + w2 + w3
        self.w1 = w1 / total_weight  # 紧凑度权重
        self.w2 = w2 / total_weight  # 贴合度权重
        self.w3 = w3 / total_weight  # 材料利用率权重
        
        self.spacing = spacing
        self.fitting_threshold = 0.25  # 贴合度阈值25%
        
        # PlacementWorker配置
        self.config = {
            'spacing': spacing,
            'clipperScale': 10000000,
            'curveTolerance': 0
        }

    def calculate_step_reward(self, 
                            bin_polygon: List[Dict], 
                            parts: List, 
                            placement_sequence: List[int], 
                            current_step: int,
                            previous_layout_info: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        计算当前步骤的即时奖励
        
        Args:
            bin_polygon: 容器多边形
            parts: 零件列表
            placement_sequence: 当前的放置序列
            current_step: 当前步骤（0-based）
            previous_layout_info: 上一步的布局信息（用于计算增量）
            
        Returns:
            reward: 当前步骤的奖励值
            layout_info: 当前布局信息（用于下一步计算）
        """
        # 获取当前要放置的零件序列
        current_sequence = placement_sequence[:current_step + 1]
        current_parts = [parts[i] for i in current_sequence]
        rotations = [0] * len(current_parts)  # 简化：不考虑旋转
        
        # 使用PlacementWorker进行实际排版
        worker = PlacementWorker(
            bin_polygon=bin_polygon,
            paths=current_parts,
            ids=current_sequence,
            rotations=rotations,
            config=self.config
        )
        
        placed, unplaced, bounding_box_area = worker.place_paths()
        
        # 计算当前布局信息
        current_layout_info = self._calculate_layout_info(placed, bounding_box_area, bin_polygon)
        
        # 计算三维奖励
        r_compactness = self._calculate_compactness_reward(current_layout_info)
        r_fitting = self._calculate_fitting_reward(placed[-1] if placed else None, placed[:-1] if len(placed) > 1 else [], bin_polygon, current_step)
        r_utilization = self._calculate_utilization_reward(current_layout_info, previous_layout_info)
        
        # 加权合成最终奖励
        total_reward = self.w1 * r_compactness + self.w2 * r_fitting + self.w3 * r_utilization
        
        # 记录详细信息
        reward_details = {
            'total_reward': total_reward,
            'compactness_reward': r_compactness,
            'fitting_reward': r_fitting,
            'utilization_reward': r_utilization,
            'layout_info': current_layout_info
        }
        
        return total_reward, reward_details

    def _calculate_layout_info(self, placed_parts: List, bounding_box_area: float, bin_polygon: List[Dict]) -> Dict:
        """计算布局信息"""
        # 计算已放置零件的总面积
        total_parts_area = sum(abs(GeometryUtil.polygon_area(part)) for part in placed_parts)
        
        # 计算容器面积
        bin_area = abs(GeometryUtil.polygon_area(bin_polygon))
        
        # 计算材料利用率
        material_utilization = total_parts_area / bounding_box_area if bounding_box_area > 0 else 0
        
        # 计算边界框占容器比例
        bbox_ratio = bounding_box_area / bin_area if bin_area > 0 else 1.0
        
        # 计算废料面积
        waste_area = bounding_box_area - total_parts_area if bounding_box_area > total_parts_area else 0
        
        return {
            'total_parts_area': total_parts_area,
            'bounding_box_area': bounding_box_area,
            'bin_area': bin_area,
            'material_utilization': material_utilization,
            'bbox_ratio': bbox_ratio,
            'waste_area': waste_area,
            'num_placed_parts': len(placed_parts)
        }

    def _calculate_compactness_reward(self, layout_info: Dict) -> float:
        """
        计算紧凑度奖励
        R_compactness(t) = (∑ᵢ₌₁ⁿ Aᵢ) / BBox_area(t)
        """
        return layout_info['material_utilization']

    def _calculate_fitting_reward(self, new_part, placed_parts: List, bin_polygon: List[Dict], current_step: int) -> float:
        """
        计算贴合度奖励
        当新零件与板材边缘及其他零件接触长度占其周长的比例超过25%时，
        奖励值等于该比例；否则给予-1惩罚
        """
        if current_step == 0 or new_part is None:
            return 0.0
        
        # 计算新零件的周长
        perimeter = self._calculate_perimeter(new_part)
        if perimeter <= 0:
            return -1.0
        
        # 计算与板材边缘的接触长度
        contact_with_bin = self._calculate_contact_with_bin(new_part, bin_polygon)
        
        # 计算与其他已排样零件的接触长度
        contact_with_parts = self._calculate_contact_with_parts(new_part, placed_parts)
        
        # 总接触长度
        total_contact_length = contact_with_bin + contact_with_parts
        
        # 计算贴合比例
        fitting_ratio = total_contact_length / perimeter
        
        # 根据阈值返回奖励
        if fitting_ratio > self.fitting_threshold:
            return fitting_ratio
        else:
            return -1.0

    def _calculate_utilization_reward(self, current_layout_info: Dict, previous_layout_info: Optional[Dict]) -> float:
        """
        计算材料利用率奖励
        R_utilization(t) = {
            +1.0,  if ΔWaste ≤ 0 (废料面积减少或不变)
            -ΔWaste/Waste_prev,  if ΔWaste > 0 (废料面积增加)
        }
        """
        if previous_layout_info is None:
            return 0.5
        
        current_waste = current_layout_info['waste_area']
        previous_waste = previous_layout_info['waste_area']
        delta_waste = current_waste - previous_waste
        
        if delta_waste <= 0:
            return 1.0
        else:
            return -delta_waste / previous_waste if previous_waste > 0 else -0.5

    def _calculate_perimeter(self, part) -> float:
        """计算零件周长"""
        if not part or len(part) < 3:
            return 0.0
        
        perimeter = 0
        for i in range(len(part)):
            p1 = part[i]
            p2 = part[(i + 1) % len(part)]
            dx = p2['x'] - p1['x']
            dy = p2['y'] - p1['y']
            perimeter += (dx**2 + dy**2)**0.5
        return perimeter

    def _calculate_contact_with_bin(self, new_part, bin_polygon: List[Dict]) -> float:
        """计算新零件与板材边缘的接触长度"""
        if not new_part or not bin_polygon:
            return 0.0
        
        contact_length = 0.0
        tolerance = 1e-3  # 接触判断的容差
        
        # 遍历新零件的每条边
        for i in range(len(new_part)):
            part_edge_start = new_part[i]
            part_edge_end = new_part[(i + 1) % len(new_part)]
            
            # 遍历板材的每条边
            for j in range(len(bin_polygon)):
                bin_edge_start = bin_polygon[j]
                bin_edge_end = bin_polygon[(j + 1) % len(bin_polygon)]
                
                # 计算两条边的重叠长度
                overlap = self._calculate_edge_overlap(part_edge_start, part_edge_end, 
                                                     bin_edge_start, bin_edge_end, tolerance)
                contact_length += overlap
        
        return contact_length
    
    def _calculate_contact_with_parts(self, new_part, placed_parts: List) -> float:
        """计算新零件与其他已排样零件的接触长度"""
        if not new_part or not placed_parts:
            return 0.0
        
        contact_length = 0.0
        tolerance = 1e-3  # 接触判断的容差
        
        # 遍历新零件的每条边
        for i in range(len(new_part)):
            part_edge_start = new_part[i]
            part_edge_end = new_part[(i + 1) % len(new_part)]
            
            # 遍历所有已放置零件
            for placed_part in placed_parts:
                if not placed_part:
                    continue
                
                # 遍历已放置零件的每条边
                for j in range(len(placed_part)):
                    placed_edge_start = placed_part[j]
                    placed_edge_end = placed_part[(j + 1) % len(placed_part)]
                    
                    # 计算两条边的重叠长度
                    overlap = self._calculate_edge_overlap(part_edge_start, part_edge_end, 
                                                         placed_edge_start, placed_edge_end, tolerance)
                    contact_length += overlap
        
        return contact_length
    
    def _calculate_edge_overlap(self, edge1_start: Dict, edge1_end: Dict, 
                               edge2_start: Dict, edge2_end: Dict, tolerance: float) -> float:
        """计算两条边的重叠长度"""
        # 简化实现：检查两条边是否平行且距离小于容差
        # 计算边的向量
        vec1 = {'x': edge1_end['x'] - edge1_start['x'], 'y': edge1_end['y'] - edge1_start['y']}
        vec2 = {'x': edge2_end['x'] - edge2_start['x'], 'y': edge2_end['y'] - edge2_start['y']}
        
        # 计算边的长度
        len1 = (vec1['x']**2 + vec1['y']**2)**0.5
        len2 = (vec2['x']**2 + vec2['y']**2)**0.5
        
        if len1 < tolerance or len2 < tolerance:
            return 0.0
        
        # 归一化向量
        vec1_norm = {'x': vec1['x'] / len1, 'y': vec1['y'] / len1}
        vec2_norm = {'x': vec2['x'] / len2, 'y': vec2['y'] / len2}
        
        # 检查是否平行
        cross_product = abs(vec1_norm['x'] * vec2_norm['y'] - vec1_norm['y'] * vec2_norm['x'])
        if cross_product > tolerance:
            return 0.0  # 不平行
        
        # 计算点到直线的距离
        dist = self._point_to_line_distance(edge1_start, edge2_start, edge2_end)
        if dist > tolerance:
            return 0.0  # 距离太远
        
        # 计算重叠长度（简化：使用投影方法）
        # 将edge1的端点投影到edge2所在直线上
        proj1 = self._project_point_on_line(edge1_start, edge2_start, edge2_end)
        proj2 = self._project_point_on_line(edge1_end, edge2_start, edge2_end)
        
        # 计算重叠区间
        overlap = self._calculate_line_segment_overlap(proj1, proj2, 0.0, len2)
        
        return max(0.0, overlap)
    
    def _point_to_line_distance(self, point: Dict, line_start: Dict, line_end: Dict) -> float:
        """计算点到直线的距离"""
        # 使用向量叉积计算点到直线的距离
        dx = line_end['x'] - line_start['x']
        dy = line_end['y'] - line_start['y']
        
        if dx == 0 and dy == 0:
            return ((point['x'] - line_start['x'])**2 + (point['y'] - line_start['y'])**2)**0.5
        
        # 计算垂直距离
        cross = abs((point['x'] - line_start['x']) * dy - (point['y'] - line_start['y']) * dx)
        line_length = (dx**2 + dy**2)**0.5
        
        return cross / line_length
    
    def _project_point_on_line(self, point: Dict, line_start: Dict, line_end: Dict) -> float:
        """将点投影到直线上，返回投影点在直线上的参数位置"""
        dx = line_end['x'] - line_start['x']
        dy = line_end['y'] - line_start['y']
        
        if dx == 0 and dy == 0:
            return 0.0
        
        # 计算投影参数
        t = ((point['x'] - line_start['x']) * dx + (point['y'] - line_start['y']) * dy) / (dx**2 + dy**2)
        
        # 将t转换为距离
        line_length = (dx**2 + dy**2)**0.5
        return t * line_length
    
    def _calculate_line_segment_overlap(self, proj1: float, proj2: float, 
                                       seg_start: float, seg_end: float) -> float:
        """计算两个线段的重叠长度"""
        # 确保proj1 <= proj2
        if proj1 > proj2:
            proj1, proj2 = proj2, proj1
        
        # 计算重叠区间
        overlap_start = max(proj1, seg_start)
        overlap_end = min(proj2, seg_end)
        
        return max(0.0, overlap_end - overlap_start)

    def update_weights(self, w1: float, w2: float, w3: float):
        """动态更新权重系数"""
        total_weight = w1 + w2 + w3
        self.w1 = w1 / total_weight
        self.w2 = w2 / total_weight
        self.w3 = w3 / total_weight

    def get_weights(self) -> Tuple[float, float, float]:
        """获取当前权重"""
        return self.w1, self.w2, self.w3

    def calculate_final_reward(self, 
                             bin_polygon: List[Dict], 
                             parts: List, 
                             placement_sequence: List[int]) -> Dict:
        """
        计算完整序列的最终奖励（用于评估）
        
        Args:
            bin_polygon: 容器多边形
            parts: 零件列表
            placement_sequence: 完整的放置序列
            
        Returns:
            final_metrics: 最终的评估指标
        """
        # 使用PlacementWorker进行完整排版
        sequence_parts = [parts[i] for i in placement_sequence]
        rotations = [0] * len(sequence_parts)
        
        worker = PlacementWorker(
            bin_polygon=bin_polygon,
            paths=sequence_parts,
            ids=placement_sequence,
            rotations=rotations,
            config=self.config
        )
        
        placed, unplaced, bounding_box_area = worker.place_paths()
        
        # 计算最终布局信息
        final_layout_info = self._calculate_layout_info(placed, bounding_box_area, bin_polygon)
        
        # 计算各维度最终得分
        final_compactness = self._calculate_compactness_reward(final_layout_info)
        final_fitting = 1.0  # 完整序列的贴合度需要特殊计算
        final_utilization = 1.0 if final_layout_info['waste_area'] == 0 else final_layout_info['material_utilization']
        
        final_score = self.w1 * final_compactness + self.w2 * final_fitting + self.w3 * final_utilization
        
        return {
            'final_score': final_score,
            'material_utilization': final_layout_info['material_utilization'],
            'compactness': final_compactness,
            'bbox_ratio': final_layout_info['bbox_ratio'],
            'waste_area': final_layout_info['waste_area'],
            'bounding_box_area': bounding_box_area,
            'total_parts_area': final_layout_info['total_parts_area'],
            'num_placed_parts': len(placed),
            'num_unplaced_parts': len(unplaced)
        } 