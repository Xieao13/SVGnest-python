import numpy as np
import torch
import json
from tqdm import tqdm
import datetime
from typing import List, Dict, Optional
import argparse
import random

from src.train_placement_improved import ImprovedPlacementNetwork
from utils.placement_worker import PlacementWorker
from utils.geometry_util import GeometryUtil


class BinPackingPredictor:
    
    def __init__(self, model_path: str, device: str = 'auto'):
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
        self._load_model()
        
    def _setup_device(self, device: str) -> torch.device:
        if device == 'auto':
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)
        return device
    
    def _load_model(self):
        self.model = ImprovedPlacementNetwork(
            input_dim=2,
            hidden_dim=256,
            seq_len=60,
            bin_dim=2
        )
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
    
    def predict_single_instance(self, instance: Dict) -> Dict:
        bin_size = instance['bin']
        parts = instance['parts']
        
        parts_tensor = torch.tensor(parts, dtype=torch.float32).unsqueeze(0)
        bin_tensor = torch.tensor(bin_size, dtype=torch.float32).unsqueeze(0)
        
        parts_tensor = parts_tensor.to(self.device)
        bin_tensor = bin_tensor.to(self.device)
        
        with torch.no_grad():
            placement_indices = self.model.predict(parts_tensor, bin_tensor)
            placement_order = placement_indices[0].cpu().numpy()
            
            num_parts = len(parts)
            placement_order = placement_order[:num_parts]
            
            rotations = np.array([random.randint(0, 3) for _ in range(num_parts)])
        
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
        
        return result
    
    def _calculate_efficiency_same_as_model(self, instance: Dict, placement_order: np.ndarray, rotations: np.ndarray) -> float:
        bin_width, bin_height = instance['bin']
        bin_polygon = [
            {'x': 0, 'y': 0},
            {'x': bin_width, 'y': 0},
            {'x': bin_width, 'y': bin_height},
            {'x': 0, 'y': bin_height},
            {'x': 0, 'y': 0}
        ]
        
        parts_polygons = []
        for i, (width, height) in enumerate(instance['parts']):
            polygon = [
                {'x': 0, 'y': 0},
                {'x': width, 'y': 0},
                {'x': width, 'y': height},
                {'x': 0, 'y': height},
                {'x': 0, 'y': 0}
            ]
            
            CustomList = type('CustomList', (list,), {})
            polygon = CustomList(polygon)
            polygon.id = i
            
            parts_polygons.append(polygon)
        
        ordered_polygons = [parts_polygons[i] for i in placement_order]
        
        rotation_degrees = rotations[placement_order] * 90
        
        worker = PlacementWorker(
            bin_polygon=bin_polygon,
            paths=ordered_polygons,
            ids=list(range(len(ordered_polygons))),
            rotations=rotation_degrees.tolist(),
            config=self.config
        )
        
        placed, unplaced, bin_area = worker.place_paths()
        
        if bin_area > 0:
            placed_area = sum(abs(GeometryUtil.polygon_area(path)) for path in placed)
            efficiency = placed_area / bin_area
            return min(efficiency, 1.0)
        else:
            return 0.0
    
    def predict_batch(self, instances: List[Dict]) -> List[Dict]:
        results = []
        
        for i, instance in enumerate(tqdm(instances, desc="预测进度")):
            result = self.predict_single_instance(instance)
            results.append(result)
        
        return results
    
    def predict_from_file(self, input_file: str, output_file: str, max_instances: Optional[int] = None):
        instances = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):   
                instance = json.loads(line.strip())
                instances.append(instance)
        
        results = self.predict_batch(instances)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

def main():
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
    
    predictor = BinPackingPredictor(
        model_path=args.model_path,
        device=args.device
    )
    
    predictor.predict_from_file(
        input_file=args.input_file,
        output_file=args.output_file,
        max_instances=args.max_instances
    )


if __name__ == '__main__':
    main() 