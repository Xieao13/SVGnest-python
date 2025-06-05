import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import logging
import sys
from tqdm import tqdm
import datetime
from typing import List, Dict, Any, Optional
import argparse
import random

# 导入必要的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_placement_improved import ImprovedPlacementNetwork, PlacementOnlyDataset, improved_collate_fn
from torch.utils.data import DataLoader

# 导入svgnest相关模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_util import GeometryUtil
from utils.placement_worker import PlacementWorker

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class BinPackingPredictor:
    """基于改进的放置顺序预测模型进行零件排布预测的类"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        初始化预测器
        
        Args:
            model_path: 训练好的模型文件路径
            device: 计算设备 ('auto', 'cuda', 'cpu', 'mps')
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.config = {
            'spacing': 0,
            'rotations': 4,
            'clipperScale': 10000000,
            'curveTolerance': 0,
            'useHoles': False,
            'exploreConcave': False
        }
        
        # 加载模型
        self._load_model()
        
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"使用设备: {device} ({torch.cuda.get_device_name(0)})")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info(f"使用设备: {device} (Apple Silicon)")
            else:
                device = torch.device("cpu")
                logger.info(f"使用设备: {device}")
        else:
            device = torch.device(device)
            logger.info(f"使用指定设备: {device}")
        
        return device
    
    def _load_model(self):
        """加载训练好的改进的放置顺序预测模型"""
        try:
            # 创建模型实例（参数需要与训练时一致）
            self.model = ImprovedPlacementNetwork(
                input_dim=2,      # 零件的宽度和高度
                hidden_dim=256,   # 隐藏层维度（与改进模型一致）
                seq_len=60,       # 最大序列长度（与train_placement_improved.py保持一致）
                bin_dim=2         # 容器的宽度和高度
            )
            
            # 加载模型权重
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"成功加载改进的放置顺序预测模型: {self.model_path}")
            
            # 打印模型参数信息
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"模型参数总数: {total_params:,}")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def predict_single_instance(self, instance: Dict) -> Dict:
        """
        使用改进的模型预测单个实例的零件排布（仅预测放置顺序，旋转角度为随机生成）
        
        Args:
            instance: 包含 'bin' 和 'parts' 的字典
            
        Returns:
            预测结果字典，包含排布顺序、旋转角度和利用率
        """
        try:
            # 提取数据
            bin_size = instance['bin']  # [width, height]
            parts = instance['parts']   # [[width, height], ...]
            
            logger.debug(f"预测实例: 容器大小={bin_size}, 零件数量={len(parts)}")
            
            # 准备模型输入
            parts_tensor = torch.tensor(parts, dtype=torch.float32).unsqueeze(0)  # [1, num_parts, 2]
            bin_tensor = torch.tensor(bin_size, dtype=torch.float32).unsqueeze(0)  # [1, 2]
            
            # 移动到设备
            parts_tensor = parts_tensor.to(self.device)
            bin_tensor = bin_tensor.to(self.device)
            
            # 模型预测（仅预测放置顺序）
            with torch.no_grad():
                # 使用模型的predict方法获取放置顺序
                placement_indices = self.model.predict(parts_tensor, bin_tensor)
                placement_order = placement_indices[0].cpu().numpy()  # 取第一个batch
                
                # 截取到实际零件数量
                num_parts = len(parts)
                placement_order = placement_order[:num_parts]
                
                # 随机生成旋转角度（因为模型不预测旋转角度）
                rotations = np.array([random.randint(0, 3) for _ in range(num_parts)])
            
            # 计算实际布局和利用率
            efficiency = self._calculate_efficiency_same_as_model(instance, placement_order, rotations)
            
            result = {
                'bin': bin_size,
                'parts': parts,
                'predicted_placement': placement_order.tolist(),
                'predicted_rotation': rotations.tolist(),
                'efficiency': efficiency,
                'timestamp': datetime.datetime.now().isoformat(),
                'model_type': 'improved_placement'
            }
            
            logger.debug(f"预测完成: 利用率={efficiency:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"预测单个实例失败: {e}")
            return {
                'bin': instance.get('bin', [0, 0]),
                'parts': instance.get('parts', []),
                'predicted_placement': [],
                'predicted_rotation': [],
                'efficiency': 0.0,
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat(),
                'model_type': 'improved_placement'
            }
    
    def _calculate_efficiency_same_as_model(self, instance: Dict, placement_order: np.ndarray, rotations: np.ndarray) -> float:
        """
        使用与模型训练时完全相同的方法计算布局利用率
        
        Args:
            instance: 原始实例数据
            placement_order: 排布顺序
            rotations: 旋转角度 (0, 1, 2, 3 对应 0°, 90°, 180°, 270°)
            
        Returns:
            利用率 (0-1之间的浮点数)
        """
        try:
            # 创建容器多边形
            bin_width, bin_height = instance['bin']
            bin_polygon = [
                {'x': 0, 'y': 0},
                {'x': bin_width, 'y': 0},
                {'x': bin_width, 'y': bin_height},
                {'x': 0, 'y': bin_height},
                {'x': 0, 'y': 0}
            ]
            
            # 创建零件多边形列表（不在这里应用旋转）
            parts_polygons = []
            for i, (width, height) in enumerate(instance['parts']):
                # 创建原始矩形多边形（不应用旋转）
                polygon = [
                    {'x': 0, 'y': 0},
                    {'x': width, 'y': 0},
                    {'x': width, 'y': height},
                    {'x': 0, 'y': height},
                    {'x': 0, 'y': 0}
                ]
                
                # 添加ID属性
                CustomList = type('CustomList', (list,), {})
                polygon = CustomList(polygon)
                polygon.id = i
                
                parts_polygons.append(polygon)
            
            # 按排布顺序重新排列
            ordered_polygons = [parts_polygons[i] for i in placement_order]
            
            # 将旋转角度从 (0,1,2,3) 映射到度数 (0°,90°,180°,270°)
            rotation_degrees = rotations[placement_order] * 90
            
            # 使用PlacementWorker进行实际布局（与模型训练完全相同的方式）
            worker = PlacementWorker(
                bin_polygon=bin_polygon,
                paths=ordered_polygons,
                ids=list(range(len(ordered_polygons))),
                rotations=rotation_degrees.tolist(),  # 传入度数而不是索引
                config=self.config
            )
            
            # 执行布局
            placed, unplaced, bin_area = worker.place_paths()
            
            # 计算利用率
            if bin_area > 0:
                placed_area = sum(abs(GeometryUtil.polygon_area(path)) for path in placed)
                efficiency = placed_area / bin_area
                
                logger.debug(f"布局结果: 已放置={len(placed)}, 未放置={len(unplaced)}, "
                           f"已放置面积={placed_area:.2f}, 容器面积={bin_area:.2f}")
                
                return min(efficiency, 1.0)  # 确保不超过100%
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"计算利用率失败: {e}, 返回0")
            return 0.0
    
    def predict_batch(self, instances: List[Dict]) -> List[Dict]:
        """
        批量预测多个实例
        
        Args:
            instances: 实例列表
            
        Returns:
            预测结果列表
        """
        results = []
        
        logger.info(f"开始批量预测，共 {len(instances)} 个实例")
        
        for i, instance in enumerate(tqdm(instances, desc="预测进度")):
            try:
                result = self.predict_single_instance(instance)
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    avg_efficiency = np.mean([r['efficiency'] for r in results[-100:]])
                    logger.info(f"已完成 {i + 1}/{len(instances)}, 最近100个平均利用率: {avg_efficiency:.2%}")
                    
            except Exception as e:
                logger.error(f"预测第 {i+1} 个实例失败: {e}")
                # 添加失败的结果
                results.append({
                    'bin': instance.get('bin', [0, 0]),
                    'parts': instance.get('parts', []),
                    'predicted_placement': [],
                    'predicted_rotation': [],
                    'efficiency': 0.0,
                    'error': str(e),
                    'timestamp': datetime.datetime.now().isoformat(),
                    'model_type': 'improved_placement'
                })
        
        # 计算总体统计
        valid_results = [r for r in results if r['efficiency'] > 0]
        if valid_results:
            avg_efficiency = np.mean([r['efficiency'] for r in valid_results])
            max_efficiency = max([r['efficiency'] for r in valid_results])
            min_efficiency = min([r['efficiency'] for r in valid_results])
            
            logger.info(f"批量预测完成:")
            logger.info(f"  有效预测: {len(valid_results)}/{len(instances)}")
            logger.info(f"  平均利用率: {avg_efficiency:.2%}")
            logger.info(f"  最高利用率: {max_efficiency:.2%}")
            logger.info(f"  最低利用率: {min_efficiency:.2%}")
        
        return results
    
    def predict_from_file(self, input_file: str, output_file: str, max_instances: Optional[int] = None):
        """
        从文件读取实例并预测，结果保存到文件
        
        Args:
            input_file: 输入JSONL文件路径
            output_file: 输出JSONL文件路径
            max_instances: 最大处理实例数量，None表示处理全部
        """
        logger.info(f"从文件预测: {input_file} -> {output_file}")
        
        # 读取输入文件
        instances = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if max_instances and len(instances) >= max_instances:
                        break
                    
                    try:
                        instance = json.loads(line.strip())
                        instances.append(instance)
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过第 {line_num+1} 行，JSON解析错误: {e}")
                        
            logger.info(f"成功读取 {len(instances)} 个实例")
            
        except Exception as e:
            logger.error(f"读取输入文件失败: {e}")
            raise
        
        # 批量预测
        results = self.predict_batch(instances)
        
        # 保存结果
        try:
            # 只有当输出文件包含目录路径时才创建目录
            output_dir = os.path.dirname(output_file)
            if output_dir:  # 如果目录不为空
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
            logger.info(f"预测结果已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"保存结果文件失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于改进的放置顺序预测模型进行零件排布预测')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的改进模型文件路径 (如: models/best_improved_placement_model.pth)')
    parser.add_argument('--input_file', type=str, required=True,
                       help='输入测试数据文件路径 (JSONL格式)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='输出预测结果文件路径 (JSONL格式)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu', 'mps'],
                       help='计算设备')
    parser.add_argument('--max_instances', type=int, default=None,
                       help='最大处理实例数量，默认处理全部')
    
    args = parser.parse_args()
    
    # 检查文件路径
    if not os.path.exists(args.model_path):
        logger.error(f"模型文件不存在: {args.model_path}")
        return
        
    if not os.path.exists(args.input_file):
        logger.error(f"输入文件不存在: {args.input_file}")
        return
    
    try:
        # 创建预测器
        predictor = BinPackingPredictor(
            model_path=args.model_path,
            device=args.device
        )
        
        # 执行预测
        predictor.predict_from_file(
            input_file=args.input_file,
            output_file=args.output_file,
            max_instances=args.max_instances
        )
        
        logger.info("预测任务完成!")
        
    except Exception as e:
        logger.error(f"预测任务失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 