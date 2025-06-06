# svgnest.py

import math
import time
import json
from typing import List, Dict, Any, Optional, Callable, Union
import pyclipper
from geometry_util import GeometryUtil
from svg_parser import SvgParser
from genetic_algorithm import GeneticAlgorithm
from placement_worker import PlacementWorker
from lxml import etree
from copy import deepcopy
import sys
import os
from tqdm import tqdm


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
            'curveTolerance': 0,
            'spacing': 0,
            'rotations': 2,
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

    def load_from_jsonl(self, instance):
        """从JSONL实例加载数据"""
        # 清空svgnest 重新初始化
        self.__init__()

        # 设置容器
        bin_width, bin_height = instance['bin']
        self.bin_bounds = {
            'x': 0,
            'y': 0,
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

        return True

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

    def set_config(self, c: Optional[Dict] = None):
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
                print(
                    f"get_parts: 处理元素 {i + 1}/{len(paths)}, 类型: {path.nodeName if hasattr(path, 'nodeName') else '未知'}")
                poly = parser.polygonify(path)
                if not poly:
                    print(f"get_parts: 元素 {i + 1} 无法转换为多边形，跳过")
                    continue

                print(f"get_parts: 多边形转换成功，点数: {len(poly)}")

                # 转换为自定义列表以支持添加属性
                if isinstance(poly, list) and not hasattr(poly, 'source'):
                    poly = CustomList(poly)

                # 清理多边形
                cleaned = self.clean_polygon(poly)
                if not cleaned:
                    print(f"get_parts: 清理多边形 {i + 1} 失败，跳过")
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
                    print(f"get_parts: 元素 {i + 1} 添加为有效零件")
                else:
                    print(f"get_parts: 元素 {i + 1} 面积太小或点数不足，跳过")
            except Exception as e:
                print(f"get_parts: 处理元素 {i + 1} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"get_parts: 找到 {len(polygons)} 个有效零件")

        if not polygons:
            print("get_parts: 没有找到有效零件，返回空列表")
            return []

        # 为每个零件分配ID
        try:
            print("get_parts: 为零件分配ID...")
            for i, poly in enumerate(polygons):
                poly.id = i
                print(f"get_parts: 分配ID {i} 给零件 {i + 1}")

            print(f"get_parts: 完成，返回 {len(polygons)} 个零件")
            return polygons
        except Exception as e:
            print(f"get_parts: 分配ID时出错: {e}")
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

            # 计算坐标范围
            x_coords = [p['x'] for p in polygon]
            y_coords = [p['y'] for p in polygon]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # 计算合适的缩放因子
            max_dim = max(x_max - x_min, y_max - y_min)
            if max_dim > 0:
                scale = min(1000000 / max_dim, self.config['clipperScale'])
            else:
                scale = self.config['clipperScale']

            # 构建 clipper 坐标
            points_tuple = [(int(p['x'] * scale),
                             int(p['y'] * scale)) for p in polygon]

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
                tolerance = int(self.config['curveTolerance'] * scale)
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
                        'x': round(p[0] / scale),
                        'y': round(p[1] / scale)
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
        """
        将布局应用到SVG文件并返回结果
        """
        from copy import deepcopy
        from xml.dom import minidom

        print("apply_placement: 开始生成SVG...")
        print(f"apply_placement: 输入数据: {placements}")

        # 创建一个新的SVG文档
        doc = minidom.Document()
        svg = doc.createElement('svg')
        svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
        svg.setAttribute('version', '1.1')
        doc.appendChild(svg)

        # 设置视图框和尺寸
        svg.setAttribute('viewBox', f"0 0 {self.bin_bounds['width']} {self.bin_bounds['height']}")
        svg.setAttribute('width', str(self.bin_bounds['width']))
        svg.setAttribute('height', str(self.bin_bounds['height']))

        # 添加容器矩形
        bin_element = doc.createElement('rect')
        bin_element.setAttribute('x', '0')
        bin_element.setAttribute('y', '0')
        bin_element.setAttribute('width', str(self.bin_bounds['width']))
        bin_element.setAttribute('height', str(self.bin_bounds['height']))
        bin_element.setAttribute('fill', 'white')
        bin_element.setAttribute('stroke', 'red')
        svg.appendChild(bin_element)

        # 验证放置结果
        if not placements:
            print("apply_placement: 无效的放置结果")
            return doc.toxml()

        # 处理放置结果
        if isinstance(placements, dict):
            # 如果是字典格式的结果
            paths = placements.get('paths', [])
            placement_info = placements.get('placements', [])
            print(f"apply_placement: 字典格式，路径数量={len(paths)}，放置信息数量={len(placement_info)}")
        else:
            # 如果是列表格式的结果
            paths = []
            placement_info = []
            for placement in placements:
                if isinstance(placement, dict):
                    path = placement.get('path', [])
                    if path:  # 只添加非空路径
                        paths.append(path)
                        placement_info.append({
                            'x': placement.get('x', 0),
                            'y': placement.get('y', 0),
                            'rotation': placement.get('rotation', 0)
                        })
            print(f"apply_placement: 列表格式，路径数量={len(paths)}，放置信息数量={len(placement_info)}")

        print(f"apply_placement: 处理 {len(paths)} 个路径")

        # 添加每个零件
        for i, path in enumerate(paths):
            if not path:  # 跳过空路径
                print(f"apply_placement: 跳过空路径 {i}")
                continue

            # 验证路径数据
            if not isinstance(path, list):
                print(f"apply_placement: 路径 {i} 格式无效")
                continue

            # 构建路径数据
            d = []
            try:
                for j, point in enumerate(path):
                    if not isinstance(point, dict) or 'x' not in point or 'y' not in point:
                        print(f"apply_placement: 路径 {i} 点 {j} 格式无效")
                        continue
                    # 计算相对于容器的坐标
                    px = point['x']
                    py = self.bin_bounds['height'] - point['y']
                    cmd = 'M' if j == 0 else 'L'
                    d.append(f"{cmd} {px},{py}")

                if not d:  # 如果没有有效点，跳过
                    print(f"apply_placement: 路径 {i} 没有有效点")
                    continue

                # 闭合路径
                d.append('Z')

                # 创建组元素
                g = doc.createElement('g')

                # 创建路径元素
                path_element = doc.createElement('path')
                path_element.setAttribute('d', ' '.join(d))
                path_element.setAttribute('fill', 'yellow')
                path_element.setAttribute('stroke', 'black')
                g.appendChild(path_element)

                # 只有当路径有内容时才添加到SVG
                svg.appendChild(g)
                print(f"apply_placement: 成功添加路径 {i}")

            except Exception as e:
                print(f"apply_placement: 处理路径 {i} 时出错: {e}")
                continue

        print("apply_placement: SVG生成完成")
        return doc.toxml()

    def _flatten_tree(self, tree: List, hole: bool) -> List:
        """展平树结构"""
        flat = []
        for node in tree:
            flat.append(node)
            node.hole = hole
            if hasattr(node, 'children') and node.children:
                flat.extend(self._flatten_tree(node.children, not hole))
        return flat

    def start(self, is_svg=True) -> bool:
        """开始布局计算"""
        if is_svg:
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

        else:
            self.tree = self.parts
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
                # 添加迭代计数器
                iteration = 0
                max_iterations = 10  # 最大迭代次数
                best_efficiency = 0

                while iteration < max_iterations and not self.working:
                    print(f"\n开始第 {iteration + 1} 次迭代...")
                    self.launch_workers(self.tree, self.bin_polygon, self.config)

                    # 更新最佳效率
                    if self.best and self.best.get('efficiency', 0) > best_efficiency:
                        best_efficiency = self.best['efficiency']
                        print(f"找到新的最佳解，效率={best_efficiency:.2%}")

                    iteration += 1

                print(f"\n迭代结束，共进行 {iteration} 次迭代")
                print(f"最终效率={best_efficiency:.2%}")
                self.working = False

        worker()  # 立即开始第一次计算
        self.GA = None
        return True

    def _offset_tree(self, tree: List, offset: float):
        """递归偏移树结构"""
        for i, path in enumerate(tree):
            offset_paths = self.polygon_offset(path, offset)
            if len(offset_paths) == 1:
                tree[i][:] = offset_paths[0]

            if hasattr(path, 'children') and path.children:
                self._offset_tree(path.children, -offset)

    def launch_workers(self, tree, bin_polygon, config):
        """启动工作进程"""
        print("launch_workers: 开始...")

        # 验证输入
        if not tree or not bin_polygon:
            print("launch_workers: 输入无效")
            return False

        # 清理容器多边形
        bin_polygon = GeometryUtil.clean_polygon(bin_polygon)
        if not bin_polygon:
            print("launch_workers: 容器多边形无效")
            return False

        # 计算容器面积
        bin_area = abs(GeometryUtil.polygon_area(bin_polygon))
        if bin_area < 1e-6:
            print("launch_workers: 容器面积过小")
            return False

        print(f"launch_workers: 容器多边形有效，面积={bin_area}")

        # 计算所有路径的总面积
        total_area = 0
        valid_paths = []
        for path in tree:
            path_area = abs(GeometryUtil.polygon_area(path))
            if path_area > 1e-6:
                total_area += path_area
                valid_paths.append(path)

        if not valid_paths:
            print("launch_workers: 没有有效的路径")
            return False

        print(f"launch_workers: 有效路径数量={len(valid_paths)}，总面积={total_area}")

        # 初始化遗传算法
        if self.GA is None:
            print("launch_workers: 初始化遗传算法...")
            # 按面积从大到小排序作为初始个体
            valid_paths.sort(key=lambda x: abs(GeometryUtil.polygon_area(x)), reverse=True)
            self.GA = GeneticAlgorithm(valid_paths, bin_polygon, config)

        # 获取当前需要评估的个体
        individual = None
        for ind in self.GA.population:
            if not hasattr(ind, 'fitness') or ind.fitness is None:
                individual = ind
                break

        if individual is None:
            # 所有个体都已评估，开始下一代
            print("launch_workers: 开始新一代...")
            self.GA.generation()
            individual = self.GA.population[1]  # 使用第二个个体开始新的一代

        # 获取当前个体的放置顺序和旋转角度
        placelist = individual.placement
        rotations = individual.rotation

        # 创建放置工作器
        try:
            worker = PlacementWorker(
                bin_polygon=bin_polygon,
                paths=placelist,
                ids=[i for i in range(len(placelist))],
                rotations=rotations,
                config=config,
            )
        except Exception as e:
            print(f"launch_workers: 创建放置工作器时出错: {e}")
            return False

        # 执行放置计算
        try:
            # 执行放置计算
            placed, unplaced, max_area = worker.place_paths()

            if not placed:
                print("launch_workers: 放置计算失败")
                return False

            # 计算性能
            placed_area = sum(abs(GeometryUtil.polygon_area(path)) for path in placed)
            efficiency = placed_area / max_area if bin_area > 0 else 0

            print(f"launch_workers: 放置完成，利用率={efficiency:.2%}")
            if efficiency >= 1:
                print(f"出错啦！{bin_polygon}，{tree}")
            print(f"launch_workers: 已放置 {len(placed)} 个路径，未放置 {len(unplaced)} 个路径")

            # 更新个体的适应度
            individual.fitness = 1 - efficiency  # 适应度越小越好

            # 更新最佳结果
            if not self.best or efficiency > self.best.get('efficiency', 0):
                # 构建结果
                result = {
                    'paths': placed,
                    'unplaced': unplaced,
                    'efficiency': efficiency
                }

                # 保存结果
                self.best = result

            return True

        except Exception as e:
            print(f"launch_workers: 执行放置计算时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def stop(self):
        """停止布局计算"""
        self.working = False
        self.worker_timer = None

    def calculate_nfp_cache(self, bin_polygon: List[Dict], paths: List[Dict]) -> Dict:
        """计算NFP缓存"""
        print("calculate_nfp_cache: 开始计算...")
        nfp_cache = {}

        # 验证输入
        if not bin_polygon or not paths:
            print("calculate_nfp_cache: 输入无效")
            return nfp_cache

        # 计算每个路径的NFP
        for i, path in enumerate(paths):
            print(f"calculate_nfp_cache: 处理路径 {i}")

            # 计算与容器的NFP
            try:
                # 尝试不同的起始点
                for offset in range(len(path)):
                    rotated_polygon = path[offset:] + path[:offset]
                    nfp = GeometryUtil.no_fit_polygon(bin_polygon, rotated_polygon, inside=True)
                    if nfp and len(nfp) >= 3:
                        key = f"bin,{i}"
                        nfp_cache[key] = nfp
                        print(f"calculate_nfp_cache: 成功计算路径 {i} 与容器的NFP")
                        break
                else:
                    print(f"calculate_nfp_cache: 路径 {i} 与容器的NFP计算失败")
            except Exception as e:
                print(f"calculate_nfp_cache: 计算路径 {i} 与容器的NFP时出错: {e}")

            # 计算与其他路径的NFP
            for j in range(i + 1, len(paths)):
                other_polygon = paths[j]
                # other_polygon = GeometryUtil.clean_polygon(paths[j])
                if not other_polygon:
                    continue

                try:
                    # 尝试不同的起始点
                    for offset in range(len(other_polygon)):
                        rotated_polygon = other_polygon[offset:] + other_polygon[:offset]
                        nfp = GeometryUtil.no_fit_polygon(path, rotated_polygon, inside=False)
                        if nfp and len(nfp) >= 3:
                            # 确保NFP闭合
                            if not (GeometryUtil.almost_equal(nfp[0]['x'], nfp[-1]['x']) and
                                    GeometryUtil.almost_equal(nfp[0]['y'], nfp[-1]['y'])):
                                nfp.append(nfp[0])

                            key = f"{i},{j}"
                            nfp_cache[key] = nfp
                            print(f"calculate_nfp_cache: 成功计算路径 {i} 和 {j} 之间的NFP")
                            break
                    else:
                        print(f"calculate_nfp_cache: 路径 {i} 和 {j} 之间的NFP计算失败")
                except Exception as e:
                    print(f"calculate_nfp_cache: 计算路径 {i} 和 {j} 之间的NFP时出错: {e}")

        print(f"calculate_nfp_cache: 完成，共计算 {len(nfp_cache)} 个NFP")
        return nfp_cache


def main(input_file: str, output_file: str):
    """主函数"""
    print(f"Processing file: {input_file}")

    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        return

    if '.svg' in input_file:

        # 读取SVG文件
        print("Reading SVG file...")
        with open(input_file, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        if not svg_content:
            print("Error: Failed to read SVG file")
            return

        # 创建SvgNest实例
        print("Initializing SvgNest...")
        nester = SvgNest()

        # 配置参数
        config = {
            'spacing': 0,
            'rotations': 4,
            'populationSize': 5,
            'mutationRate': 50,
            'useHoles': True,
            'exploreConcave': False
        }
        print(f"Configuring with parameters: {config}")
        nester.set_config(config)

        try:
            # ====================================解析SVG=====================================
            print("Parsing SVG content...")
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

            # =================================开始布局计算===================================
            print("Starting placement calculation...")

            result = nester.start()
            if result:
                print("Placement calculation started successfully")
            else:
                print("Error: Failed to start placement calculation")
                return

            # ==================================生成svg结果===================================
            print("Creating output directory: output")
            os.makedirs("output", exist_ok=True)

            print("Generating placement results...")
            try:
                if nester.best:
                    # 将完整的结果传递给apply_placement
                    output_svg = nester.apply_placement(nester.best)
                    output_file = os.path.join("output", "placement.svg")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(output_svg)
                    print(f"Generated placement results in {output_file}")
                else:
                    print("No valid placement results found")
            except Exception as e:
                print(f"Error generating placement results: {e}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Program completed")
    elif '.jsonl' in input_file:
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

        # 读取并处理每个实例
        with open(input_file, 'r') as f_in, open(output_file, 'a') as f_out:
            lines = f_in.readlines()
            for line in tqdm(lines, total=len(lines), desc="Processing instances"):
                instance = json.loads(line)
                print(f"Processing instance with bin size: {instance['bin']}")

                # 加载实例数据
                if not nester.load_from_jsonl(instance):
                    print("Error: Failed to load instance data")
                    continue

                nester.set_config(config)
                # 开始布局计算
                print("Starting placement calculation...")
                result = nester.start(is_svg=False)
                if not result:
                    print("Error: Failed to start placement calculation")
                    continue

                # 获取最佳结果
                if nester.best:
                    # 提取零件顺序
                    placement_order = [p.id for p in nester.best['paths']]
                    placement_rotation = [p.rotation for p in nester.best['paths']]
                    result = {
                        'bin': instance['bin'],
                        'parts': instance['parts'],
                        'placement_order': placement_order,
                        'rotation': placement_rotation,
                        'efficiency': nester.best['efficiency']
                    }
                    # 写入结果
                    f_out.write(json.dumps(result) + '\n')
                    print(f"Instance processed successfully, efficiency: {nester.best['efficiency']:.2%}")
                else:
                    print("No valid placement results found")


if __name__ == '__main__':
    svg_file = "input.svg"
    # jsonl_file = "output/instances.jsonl"
    output_file = "output/placement—0412.jsonl"
    # main(jsonl_file, output_file)
    main(svg_file, output_file)
