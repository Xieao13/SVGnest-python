# svgnest.py

import math
import time
from typing import List, Dict, Any, Optional, Callable, Union
import pyclipper
from geometry_util import GeometryUtil
from svg_parser import SvgParser
from genetic_algorithm import GeneticAlgorithm
from placement_worker import PlacementWorker

class SvgNest:
    def __init__(self):
        """初始化SvgNest"""
        self.svg = None
        self.style = None
        self.parts = None
        self.tree = None
        self.bin = None
        self.bin_polygon = None
        self.bin_bounds = None
        self.nfp_cache = {}
        
        # 配置参数
        self._config = {
            'clipperScale': 10000000,
            'curveTolerance': 0.3,
            'spacing': 0,
            'rotations': 4,
            'populationSize': 10,
            'mutationRate': 10,
            'useHoles': False,
            'exploreConcave': False
        }
        
        self.working = False
        self.GA = None
        self.best = None
        self.worker_timer = None
        self.progress = 0
    
    @property
    def config(self):
        """配置属性的getter"""
        return self._config
   
    def parse_svg(self, svg_string: str):
        """解析SVG字符串"""
        # 重置进度
        self.stop()
        
        self.bin = None
        self.bin_polygon = None
        self.tree = None
        
        print("Starting SVG parsing...")
        
        # 解析SVG
        try:
            parser = SvgParser()
            print("Created SvgParser instance")
            
            self.svg = parser.load(svg_string)
            print("SVG loaded successfully")
            
            self.style = parser.get_style()
            print(f"Style retrieved: {self.style is not None}")
            
            print("About to clean SVG...")
            self.svg = parser.clean()
            print("SVG cleaned successfully")
            
            if not self.svg:
                print("Error: SVG is None after cleaning")
                return None
                
            print("About to get parts...")
            self.tree = self.get_parts(self.svg.childNodes)
            print(f"Got {len(self.tree) if self.tree else 0} parts")
        
            print("SVG content:")
            print(self.svg.toxml())
            
            return self.svg
        except Exception as e:
            print(f"Error in parse_svg: {e}")
            import traceback
            traceback.print_exc()
            return None

    def set_bin(self, element):
        """设置容器元素"""
        if not self.svg:
            return
        self.bin = element

    def set_config(self, c: Optional[Dict] = None) -> Dict:
        """配置参数"""
        if not c:
            return self._config
            
        if c.get('curveTolerance') and not GeometryUtil.almost_equal(float(c['curveTolerance']), 0):
            self._config['curveTolerance'] = float(c['curveTolerance'])
            
        if 'spacing' in c:
            self._config['spacing'] = float(c['spacing'])
            
        if c.get('rotations') and int(c['rotations']) > 0:
            self._config['rotations'] = int(c['rotations'])
            
        if c.get('populationSize') and int(c['populationSize']) > 2:
            self._config['populationSize'] = int(c['populationSize'])
            
        if c.get('mutationRate') and int(c['mutationRate']) > 0:
            self._config['mutationRate'] = int(c['mutationRate'])
            
        if 'useHoles' in c:
            self._config['useHoles'] = bool(c['useHoles'])
            
        if 'exploreConcave' in c:
            self._config['exploreConcave'] = bool(c['exploreConcave'])
            
        parser = SvgParser()
        parser.config({'tolerance': self._config['curveTolerance']})
        
        self.best = None
        self.nfp_cache = {}
        self.bin_polygon = None
        self.GA = None
        
        return self._config

    def get_parts(self, paths: List) -> List:
        """获取零件列表"""
        print("get_parts: 开始处理零件...")
        polygons = []
        parser = SvgParser()
        
        # 创建自定义列表类
        CustomList = type('CustomList', (list,), {})
        
        # 转换所有路径为多边形
        for i, path in enumerate(paths):
            try:
                print(f"get_parts: 处理元素 {i+1}/{len(paths)}, 类型: {path.nodeName if hasattr(path, 'nodeName') else '未知'}")
                poly = parser.polygonify(path)
                if not poly:
                    print(f"get_parts: 元素 {i+1} 无法转换为多边形，跳过")
                    continue
                    
                print(f"get_parts: 多边形转换成功，点数: {len(poly)}")
                
                # 转换为自定义列表以支持添加属性
                if isinstance(poly, list) and not hasattr(poly, 'source'):
                    poly = CustomList(poly)
                
                # 清理多边形
                cleaned = self.clean_polygon(poly)
                if not cleaned:
                    print(f"get_parts: 清理多边形 {i+1} 失败，跳过")
                    continue
                
                # 确保结果是可扩展的列表
                if isinstance(cleaned, list) and not hasattr(cleaned, 'source'):
                    cleaned = CustomList(cleaned)
                
                poly = cleaned
                print(f"get_parts: 多边形清理成功，点数: {len(poly)}")
                
                # 检查面积
                if (len(poly) > 2 and 
                    abs(GeometryUtil.polygon_area(poly)) > 
                    self.config['curveTolerance'] * self.config['curveTolerance']):
                    poly.source = i
                    polygons.append(poly)
                    print(f"get_parts: 元素 {i+1} 添加为有效零件")
                else:
                    print(f"get_parts: 元素 {i+1} 面积太小或点数不足，跳过")
            except Exception as e:
                print(f"get_parts: 处理元素 {i+1} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        print(f"get_parts: 找到 {len(polygons)} 个有效零件")
        
        if not polygons:
            print("get_parts: 没有找到有效零件，返回空列表")
            return []
            
        # 将列表转换为树结构
        try:
            print("get_parts: 开始构建树结构...")
            parents = []
            id_counter = 0
            
            # 为每个叶子分配唯一ID
            for i, p1 in enumerate(polygons):
                print(f"get_parts: 检查零件 {i+1} 的层次关系")
                is_child = False
                for j, p2 in enumerate(polygons):
                    if i == j:
                        continue
                    if GeometryUtil.point_in_polygon(p1[0], p2):
                        if not hasattr(p2, 'children'):
                            p2.children = []
                        p2.children.append(p1)
                        p1.parent = p2
                        is_child = True
                        print(f"get_parts: 零件 {i+1} 是零件 {j+1} 的子元素")
                        break
                if not is_child:
                    parents.append(p1)
                    print(f"get_parts: 零件 {i+1} 是顶层元素")
                    
            # 移除非父节点
            for i in range(len(polygons)-1, -1, -1):
                if polygons[i] not in parents:
                    polygons.pop(i)
                    
            # 分配ID
            for parent in parents:
                parent.id = id_counter
                id_counter += 1
                print(f"get_parts: 分配ID {parent.id} 给顶层零件")
            
            print(f"get_parts: 树结构构建完成，返回 {len(parents)} 个顶层零件")
            return parents
        except Exception as e:
            print(f"get_parts: 构建树结构时出错: {e}")
            import traceback
            traceback.print_exc()
            return []

    def clean_polygon(self, polygon: List) -> Optional[List]:
        """清理多边形"""
        if not polygon:
            return None
    
        try:
            print(f"clean_polygon: 开始清理多边形，点数: {len(polygon)}")
            
            # 检查数据格式
            for i, p in enumerate(polygon):
                if 'x' not in p or 'y' not in p:
                    print(f"clean_polygon: 点 {i} 格式错误，跳过")
                    return None
            
            # 构建 clipper 坐标
            points_tuple = [(int(p['x'] * self.config['clipperScale']), 
                            int(p['y'] * self.config['clipperScale'])) for p in polygon]
            
            # 检查多边形是否闭合
            if (len(polygon) > 2 and 
                (polygon[0]['x'] != polygon[-1]['x'] or polygon[0]['y'] != polygon[-1]['y'])):
                # 添加闭合点
                points_tuple.append(points_tuple[0])
                print("clean_polygon: 添加闭合点")
                
            print(f"clean_polygon: 转换为clipper坐标，点数: {len(points_tuple)}")
            
            # 移除自交点
            try:
                simple = pyclipper.SimplifyPolygon(points_tuple, pyclipper.PFT_NONZERO)
                print(f"clean_polygon: 简化多边形成功，得到 {len(simple)} 个多边形")
                
                if not simple:
                    print("clean_polygon: 简化后无有效多边形")
                    return None
                    
                # 找到最大的多边形
                biggest = simple[0]
                biggest_area = abs(pyclipper.Area(biggest))
                for poly in simple[1:]:
                    area = abs(pyclipper.Area(poly))
                    if area > biggest_area:
                        biggest = poly
                        biggest_area = area
                        
                print(f"clean_polygon: 找到最大多边形，面积: {biggest_area}, 点数: {len(biggest)}")
                
                # 清理奇异点、重合点和边
                tolerance = int(self.config['curveTolerance'] * self.config['clipperScale'])
                if tolerance < 1:
                    tolerance = 1  # 确保容差至少为1
                    
                clean = pyclipper.CleanPolygon(biggest, tolerance)
                
                if not clean:
                    print("clean_polygon: 清理后无有效多边形")
                    return None
                    
                print(f"clean_polygon: 清理多边形成功，点数: {len(clean)}")
                
                # 转回原始坐标
                result = []
                for p in clean:
                    result.append({
                        'x': p[0] / self.config['clipperScale'],
                        'y': p[1] / self.config['clipperScale']
                    })
                    
                print(f"clean_polygon: 转换回原始坐标成功，点数: {len(result)}")
                
                # 复制原始多边形的属性
                if hasattr(polygon, 'source'):
                    result = type('CustomList', (list,), {})(result)
                    result.source = polygon.source
                
                # 创建一个自定义列表对象
                if isinstance(result, list) and not hasattr(result, 'source'):
                    result = type('CustomList', (list,), {})(result)
                
                return result
                
            except Exception as e:
                print(f"clean_polygon: 处理多边形时出错: {e}")
                import traceback
                traceback.print_exc()
                return None
            
        except Exception as e:
            print(f"clean_polygon: 清理多边形时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def polygon_offset(self, polygon: List, offset: float) -> List[List]:
        """多边形偏移"""
        if not offset or offset == 0:
            return [polygon]
            
        # 使用pyclipper进行偏移
        pc = pyclipper.PyclipperOffset()
        scaled_poly = pyclipper.scale_to_clipper(polygon, self.config['clipperScale'])
        pc.AddPath(scaled_poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        
        # 执行偏移
        solution = pc.Execute(offset * self.config['clipperScale'])
        
        # 转回原始坐标
        result = []
        for path in solution:
            result.append(pyclipper.scale_from_clipper(path, self.config['clipperScale']))
            
        return result

    def apply_placement(self, placements):
        """应用布局结果，生成最终SVG"""
        if not placements or not self.svg:
            return []
            
        result = []
        
        # 创建一个SVG文档副本
        for i in range(len(placements)):
            # 创建新的SVG文档
            svg_copy = self.svg.cloneNode(True)
            
            # 清除所有现有元素
            for child in list(svg_copy.childNodes):
                svg_copy.removeChild(child)
                
            # 添加容器元素
            if self.bin:
                bin_copy = self.bin.cloneNode(True)
                svg_copy.appendChild(bin_copy)
                
            # 应用每个放置的零件
            for placement in placements[i]:
                part_id = placement['id']
                part = None
                
                # 找到对应的部件
                for p in self.tree:
                    if hasattr(p, 'id') and p.id == part_id:
                        part = p
                        break
                        
                if not part:
                    continue
                    
                # 找到原始SVG元素
                if hasattr(part, 'source') and part.source >= 0 and part.source < len(self.parts):
                    original = self.parts[part.source]
                    
                    # 复制元素
                    part_copy = original.cloneNode(True)
                    
                    # 应用变换
                    if 'rotation' in placement and placement['rotation'] != 0:
                        transform = f"rotate({placement['rotation']}"
                        
                        if 'x' in placement and 'y' in placement:
                            transform += f" {placement['x']} {placement['y']}"
                            
                        transform += ")"
                        
                        # 添加变换属性
                        part_copy.setAttribute('transform', transform)
                        
                    # 添加到SVG
                    svg_copy.appendChild(part_copy)
                    
            result.append(svg_copy)
                
        return result

    def _flatten_tree(self, tree: List, hole: bool) -> List:
        """展平树结构"""
        flat = []
        for node in tree:
            flat.append(node)
            node.hole = hole
            if hasattr(node, 'children') and node.children:
                flat.extend(self._flatten_tree(node.children, not hole))
        return flat

    def start(self, progress_callback: Callable, display_callback: Callable) -> bool:
        """开始布局计算"""
        if not self.svg or not self.bin:
            return False
        
        # 设置工作状态为True
        self.working = True
        
        self.parts = list(self.svg.childNodes)
        try:
            bin_index = self.parts.index(self.bin)
            self.parts.pop(bin_index)
        except ValueError:
            pass
            
        # 构建不含bin的树
        self.tree = self.get_parts(self.parts[:])
        
        # 偏移处理
        self._offset_tree(self.tree, 0.5 * self.config['spacing'])
        
        # 处理容器多边形
        parser = SvgParser()
        self.bin_polygon = parser.polygonify(self.bin)
        self.bin_polygon = self.clean_polygon(self.bin_polygon)
        
        if not self.bin_polygon or len(self.bin_polygon) < 3:
            return False
            
        self.bin_bounds = GeometryUtil.get_polygon_bounds(self.bin_polygon)
        
        # 处理间距
        if self.config['spacing'] > 0:
            offset_bin = self.polygon_offset(self.bin_polygon, -0.5 * self.config['spacing'])
            if len(offset_bin) == 1:
                self.bin_polygon = offset_bin[0]
                
        if isinstance(self.bin_polygon, list) and not hasattr(self.bin_polygon, 'id'):
            # 创建可扩展的列表
            CustomList = type('CustomList', (list,), {})
            self.bin_polygon = CustomList(self.bin_polygon)
        self.bin_polygon.id = -1
        
        # 移动到原点
        bounds = GeometryUtil.get_polygon_bounds(self.bin_polygon)
        for point in self.bin_polygon:
            point['x'] -= bounds['x']
            point['y'] -= bounds['y']
            
        if isinstance(self.bin_polygon, list) and not hasattr(self.bin_polygon, 'width'):
            CustomList = type('CustomList', (list,), {})
            self.bin_polygon = CustomList(self.bin_polygon)
        self.bin_polygon.width = bounds['width']
        self.bin_polygon.height = bounds['height']
        
        # 确保逆时针方向
        if GeometryUtil.polygon_area(self.bin_polygon) > 0:
            self.bin_polygon.reverse()
            
        # 处理所有路径
        for path in self.tree:
            # 移除重复端点
            while (len(path) > 1 and 
                   GeometryUtil.almost_equal(path[0]['x'], path[-1]['x']) and 
                   GeometryUtil.almost_equal(path[0]['y'], path[-1]['y'])):
                path.pop()
                
            # 确保逆时针方向
            if GeometryUtil.polygon_area(path) > 0:
                path.reverse()
                
        self.working = False
        
        # 启动工作进程
        def worker():
            if not self.working:
                self.launch_workers(self.tree, self.bin_polygon, self.config,
                                 progress_callback, display_callback)
                self.working = True
            progress_callback(self.progress)
            
        self.worker_timer = worker
        worker()  # 立即开始第一次计算
        
        return True

    def _offset_tree(self, tree: List, offset: float):
        """递归偏移树结构"""
        for i, path in enumerate(tree):
            offset_paths = self.polygon_offset(path, offset)
            if len(offset_paths) == 1:
                tree[i][:] = offset_paths[0]
                
            if hasattr(path, 'children') and path.children:
                self._offset_tree(path.children, -offset)
    def launch_workers(self, tree, bin_polygon, config, progress_callback, display_callback):
        """启动工作进程，处理零件布局计算"""
        print("SvgNest.launch_workers: 开始布局计算...")
        
        # 添加调试信息
        print(f"零件数量: {len(tree) if tree else 0}")
        print(f"容器多边形有 {len(bin_polygon) if bin_polygon else 0} 个点")
        
        try:
            # 准备数据
            paths = []
            ids = []
            rotations = []
            
            # 将树形结构转换为平面列表
            for i, part in enumerate(tree):
                if hasattr(part, 'id') and part.id >= 0:
                    # 尝试各种可能的旋转角度
                    for rotation in range(self.config['rotations']):
                        angle = rotation * (360 / self.config['rotations'])
                        
                        # 创建自定义列表
                        CustomList = type('CustomList', (list,), {})
                        r = CustomList(GeometryUtil.rotate_polygon(part, angle))
                        
                        # 复制属性
                        if hasattr(part, 'source'):
                            r.source = part.source
                        else:
                            r.source = part.id  # 如果没有source属性，使用id作为备用
                            
                        r.id = part.id
                        r.rotation = angle
                        
                        paths.append(r)
                        ids.append(part.id)
                        rotations.append(angle)
                        
                        # 预先计算NFP
                        key = str({
                            'A': -1,
                            'B': part.id,
                            'inside': True,
                            'Arotation': 0,
                            'Brotation': angle
                        })
                        
                        if key not in self.nfp_cache:
                            self.nfp_cache[key] = GeometryUtil.no_fit_polygon(
                                bin_polygon, r, True, self.config['exploreConcave']
                            )
                        
                        # 为每个已存在的部件计算NFP
                        for j, placed in enumerate(tree):
                            if i != j and hasattr(placed, 'id') and placed.id >= 0:
                                for placed_rotation in range(self.config['rotations']):
                                    placed_angle = placed_rotation * (360 / self.config['rotations'])
                                    
                                    # 计算NFP
                                    key = str({
                                        'A': placed.id,
                                        'B': part.id,
                                        'inside': False,
                                        'Arotation': placed_angle,
                                        'Brotation': angle
                                    })
                                    
                                    if key not in self.nfp_cache:
                                        placed_custom = CustomList(GeometryUtil.rotate_polygon(placed, placed_angle))
                                        self.nfp_cache[key] = GeometryUtil.no_fit_polygon(
                                            placed_custom, r, False, self.config['exploreConcave']
                                        )
            
            # 如果没有零件可放置，直接返回
            if not paths:
                print("SvgNest.launch_workers: 没有零件可放置")
                self.working = False
                return
                
            # 创建遗传算法实例
            print("SvgNest.launch_workers: 初始化遗传算法...")
            self.GA = GeneticAlgorithm(paths, bin_polygon, self.config)
            
            # 初始布局
            worker = PlacementWorker(
                bin_polygon, 
                self.GA.population[0]['placement'], 
                ids, 
                self.GA.population[0]['rotation'], 
                self.config, 
                self.nfp_cache
            )
            
            # 计算布局
            print("SvgNest.launch_workers: 计算初始布局...")
            result = worker.place_paths(self.GA.population[0]['placement'])
            
            # 处理结果
            if result:
                self.GA.population[0]['fitness'] = result['fitness']
                self.GA.population[0]['placements'] = result['placements']
                
                # 触发回调
                if display_callback:
                    area = bin_polygon.width * bin_polygon.height
                    efficiency = 0
                    
                    # 计算利用率
                    for placements in self.GA.population[0]['placements']:
                        for placement in placements:
                            placement_id = placement.id if hasattr(placement, 'id') else (
                                placement['id'] if isinstance(placement, dict) and 'id' in placement else 0
                            )
                            # 确保placement_id是整数类型
                            if isinstance(placement_id, str):
                                try:
                                    placement_id = int(placement_id)
                                except ValueError:
                                    placement_id = 0
                            part_poly = tree[placement_id]
                            efficiency += abs(GeometryUtil.polygon_area(part_poly))
                    
                    efficiency /= area
                    self.best = self.GA.population[0]
                    
                    # 显示结果
                    display_callback(
                        result, 
                        efficiency, 
                        len(tree) - len(result['paths']), 
                        len(tree)
                    )
            
            print("SvgNest.launch_workers: 初始布局完成")
            self.progress = 0.5
            progress_callback(self.progress)
            
            # 继续优化
            iteration = 0
            max_iterations = 10
            while self.working and iteration < max_iterations:
                print(f"SvgNest.launch_workers: 开始迭代 {iteration + 1}/{max_iterations}")
                
                # 产生新一代
                self.GA.generation()
                
                # 为每个个体计算适应度
                for i in range(len(self.GA.population)):
                    if 'fitness' not in self.GA.population[i]:
                        worker = PlacementWorker(
                            bin_polygon, 
                            self.GA.population[i]['placement'], 
                            ids, 
                            self.GA.population[i]['rotation'], 
                            self.config, 
                            self.nfp_cache
                        )
                        
                        result = worker.place_paths(self.GA.population[i]['placement'])
                        
                        if result:
                            self.GA.population[i]['fitness'] = result['fitness']
                            self.GA.population[i]['placements'] = result['placements']
                
                # 排序找出最优解
                self.GA.population.sort(key=lambda x: x.get('fitness', float('inf')))
                self.best = self.GA.population[0]
                
                # 计算进度
                self.progress = 0.5 + (iteration / max_iterations) * 0.5
                progress_callback(self.progress)
                
                # 计算并显示最优解的效率
                if display_callback and self.best and 'placements' in self.best:
                    area = bin_polygon.width * bin_polygon.height
                    efficiency = 0
                    
                    for placements in self.best['placements']:
                        for placement in placements:
                            placement_id = placement.id if hasattr(placement, 'id') else (
                                placement['id'] if isinstance(placement, dict) and 'id' in placement else 0
                            )
                            # 确保placement_id是整数类型
                            if isinstance(placement_id, str):
                                try:
                                    placement_id = int(placement_id)
                                except ValueError:
                                    placement_id = 0
                            part_poly = tree[placement_id]
                            efficiency += abs(GeometryUtil.polygon_area(part_poly))
                    
                    efficiency /= area
                    placed = 0
                    
                    for placements in self.best['placements']:
                        placed += len(placements)
                    
                    display_callback(result, efficiency, placed, len(tree))
                
                iteration += 1
            
            print("SvgNest.launch_workers: 布局计算完成")
            self.working = False
            
        except Exception as e:
            print(f"SvgNest.launch_workers: 发生错误: {e}")
            import traceback
            traceback.print_exc()
            self.working = False
    def stop(self):
        """停止布局计算"""
        self.working = False
        self.worker_timer = None

# 在 svgnest.py 文件末尾添加以下代码

    def read_svg_file(file_path: str) -> str:
        """读取SVG文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading SVG file: {e}")
            return ""

    def progress_callback(progress: float):
        """进度回调函数"""
        print(f"Progress: {progress * 100:.2f}%")

    def display_callback(result=None, efficiency=None, placed=None, total=None):
        """显示回调函数"""
        if result:
            print(f"Placement efficiency: {efficiency * 100:.2f}%")
            print(f"Placed parts: {placed}/{total}")

def main():
    """主函数"""
    import sys
    import os
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("Usage: python svgnest.py <input.svg>")
        return
        
    input_file = sys.argv[1]
    print(f"Processing file: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        return
        
    # 读取SVG文件
    print("Reading SVG file...")
    svg_content = read_svg_file(input_file)
    if not svg_content:
        print("Error: Failed to read SVG file")
        return
    print(f"SVG file loaded, content length: {len(svg_content)} characters")
        
    # 创建SvgNest实例
    print("Initializing SvgNest...")
    nester = SvgNest()
    
    # 配置参数
    config = {
        'spacing': 0,
        'rotations': 4,
        'populationSize': 10,
        'mutationRate': 10,
        'useHoles': True,
        'exploreConcave': False
    }
    print(f"Configuring with parameters: {config}")
    nester.set_config(config)
    
    # 解析SVG
    print("Parsing SVG content...")
    try:
        svg = nester.parse_svg(svg_content)
        if not svg:
            print("Error: Failed to parse SVG - svg is None")
            return
        print("SVG parsed successfully")
            
        # 找到第一个矩形作为容器
        print("Looking for container rectangle...")
        bin_element = None
        rect_elements = svg.getElementsByTagName('rect')
        print(f"Found {len(rect_elements)} rectangle elements")
        
        if rect_elements:
            bin_element = rect_elements[0]
            print(f"Container rectangle found: {bin_element.toxml()}")
        
        if not bin_element:
            print("Error: No container rectangle found in SVG")
            return
            
        # 设置容器
        print("Setting container...")
        nester.set_bin(bin_element)
        
        # 开始布局计算
        print("Starting placement calculation...")
        result = nester.start(progress_callback, display_callback)
        if not result:
            print("Error: Failed to start placement - invalid input")
            return
            
        print("Placement calculation started successfully")
        
        # 等待计算完成
        max_wait_time = 60  # 最长等待时间（秒）
        wait_start = time.time()
        while nester.working and (time.time() - wait_start < max_wait_time):
            time.sleep(0.1)
            
        if nester.working:
            print("Warning: Calculation timeout, stopping...")
            nester.stop()
        
        # 保存结果
        output_dir = "output"
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取最终布局结果
        if nester.best:
            print("Generating placement results...")
            try:
                placements = nester.apply_placement(nester.best.placements)
                for i, placement in enumerate(placements):
                    output_file = os.path.join(output_dir, f"result_{i+1}.svg")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(placement.toxml())
                    print(f"Saved result to {output_file}")
            except Exception as e:
                print(f"Error generating placement results: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No valid placement found")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        
    print("Program completed")

def progress_callback(progress: float):
    """进度回调函数"""
    print(f"\rProgress: {progress * 100:.2f}%", end='', flush=True)

def display_callback(result=None, efficiency=None, placed=None, total=None):
    """显示回调函数"""
    if result:
        print(f"\nPlacement efficiency: {efficiency * 100:.2f}%")
        print(f"Placed parts: {placed}/{total}")
    else:
        print("\nCalculating placement...", flush=True)

def read_svg_file(file_path: str) -> str:
    """读取SVG文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Successfully read {len(content)} characters from {file_path}")
            return content
    except Exception as e:
        print(f"Error reading SVG file: {e}")
        return ""

if __name__ == "__main__":
    try:
        print("Starting SvgNest...")
        main()
        print("Program completed")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()