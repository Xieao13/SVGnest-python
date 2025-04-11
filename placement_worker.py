# placement_worker.py

import math
from typing import List, Dict, Any, Optional, Union, Tuple
import pyclipper
from geometry_util import GeometryUtil


def to_clipper_coordinates(polygon: List[Dict]) -> List[tuple]:
    """将普通坐标转换为Clipper坐标(元组格式)
    
    Args:
        polygon: 包含x,y坐标的多边形列表
        
    Returns:
        转换后的Clipper坐标元组列表
    """
    try:
        result = []
        if not polygon:
            print("警告: 输入多边形为空")
            return []

        # 定义缩放因子
        SCALE = 1000000

        # 如果输入是单个点，直接返回该点
        if len(polygon) == 1:
            point = polygon[0]
            try:
                if isinstance(point, dict):
                    x = float(point.get('x', point.get('X', 0)))
                    y = float(point.get('y', point.get('Y', 0)))
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    x = float(point[0])
                    y = float(point[1])
                elif hasattr(point, 'x') and hasattr(point, 'y'):
                    x = float(getattr(point, 'x'))
                    y = float(getattr(point, 'y'))
                else:
                    print(f"警告: 无效的点格式: {point}")
                    return []

                if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                    print(f"警告: 坐标不是有效数字: x={x}, y={y}")
                    return []

                if abs(x) > 1e6 or abs(y) > 1e6:
                    print(f"警告: 原始坐标值过大: x={x}, y={y}")
                    return []

                x = int(round(x * SCALE))
                y = int(round(y * SCALE))
                return [(x, y)]
            except (ValueError, TypeError) as e:
                print(f"警告: 单点坐标转换失败: {e}")
                return []

        # 检查输入多边形是否闭合
        if len(polygon) >= 2:
            first_point = polygon[0]
            last_point = polygon[-1]
            if isinstance(first_point, dict) and isinstance(last_point, dict):
                if not (GeometryUtil.almost_equal(first_point.get('x', 0), last_point.get('x', 0)) and
                        GeometryUtil.almost_equal(first_point.get('y', 0), last_point.get('y', 0))):
                    print("警告: 输入多边形未闭合，将自动闭合")
                    polygon = polygon + [polygon[0]]

        for point in polygon:
            # 处理字典格式的点
            if isinstance(point, dict):
                # 检查是否有x,y键
                if 'x' in point and 'y' in point:
                    try:
                        x = float(point['x'])
                        y = float(point['y'])
                    except (ValueError, TypeError):
                        print(f"警告: 坐标值无法转换为浮点数: {point}")
                        continue
                # 检查是否有X,Y键（大写）
                elif 'X' in point and 'Y' in point:
                    try:
                        x = float(point['X'])
                        y = float(point['Y'])
                    except (ValueError, TypeError):
                        print(f"警告: 坐标值无法转换为浮点数: {point}")
                        continue
                else:
                    print(f"警告: 点缺少x或y坐标: {point}")
                    continue
            # 处理列表或元组格式的点
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    x = float(point[0])
                    y = float(point[1])
                except (ValueError, TypeError):
                    print(f"警告: 坐标值无法转换为浮点数: {point}")
                    continue
            # 处理自定义对象格式的点
            elif hasattr(point, 'x') and hasattr(point, 'y'):
                try:
                    x = float(getattr(point, 'x'))
                    y = float(getattr(point, 'y'))
                except (ValueError, TypeError):
                    print(f"警告: 坐标值无法转换为浮点数: {point}")
                    continue
            else:
                print(f"警告: 无效的点格式: {point}")
                continue

            # 检查坐标是否为有效数字
            if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                print(f"警告: 坐标不是有效数字: x={x}, y={y}")
                continue

            # 检查原始坐标是否在合理范围内
            if abs(x) > 1e6 or abs(y) > 1e6:
                print(f"警告: 原始坐标值过大: x={x}, y={y}")
                continue

            # 转换到Clipper坐标（放大并取整）
            try:
                x = int(round(x * SCALE))
                y = int(round(y * SCALE))
                result.append((x, y))
            except (ValueError, TypeError, OverflowError) as e:
                print(f"警告: 坐标转换失败: {e}")
                continue

        # 验证结果
        if len(result) < 3 and len(polygon) >= 3:
            print(f"警告: 转换后的点数不足: {len(result)}")
            return []

        # 确保多边形闭合
        if len(result) >= 3 and not (GeometryUtil.almost_equal(result[0][0], result[-1][0], 0.1) and
                                     GeometryUtil.almost_equal(result[0][1], result[-1][1], 0.1)):
            result.append(result[0])

        # 验证多边形面积
        if len(result) >= 3:
            try:
                area = abs(pyclipper.Area(result))
                if area < SCALE * 0.1:  # 降低面积阈值
                    print(f"警告: 转换后的多边形面积过小: {area}")
                    return []
            except Exception as e:
                print(f"警告: 计算多边形面积时出错: {e}")
                return []

        return result
    except Exception as e:
        print(f"转换到Clipper坐标时出错: {e}")
        return []


def to_nest_coordinates(polygon: List[tuple], scale: float) -> List[Dict]:
    """将Clipper坐标转回普通坐标
    
    Args:
        polygon: Clipper坐标多边形
        scale: 缩放因子
        
    Returns:
        转换后的普通坐标列表
    """
    try:
        result = []
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                result.append({
                    'x': float(point[0]) / scale,
                    'y': float(point[1]) / scale
                })
            elif isinstance(point, dict):
                if 'X' in point and 'Y' in point:
                    result.append({
                        'x': float(point['X']) / scale,
                        'y': float(point['Y']) / scale
                    })
                elif 'x' in point and 'y' in point:
                    result.append({
                        'x': float(point['x']) / scale,
                        'y': float(point['y']) / scale
                    })
            else:
                print(f"警告: 无效的点格式: {point}")
        return result
    except Exception as e:
        print(f"转换回普通坐标时出错: {e}")
        return []


def rotate_polygon(polygon, degrees):
    """旋转多边形
    Args:
        polygon: 要旋转的多边形，可以是字典列表或自定义对象
        degrees: 旋转角度（度）
    Returns:
        旋转后的多边形
    """
    # 创建自定义列表类型
    CustomList = type('CustomList', (list,), {})
    rotated = CustomList()

    # 转换角度为弧度
    angle = math.radians(degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # 旋转每个点
    for point in polygon:
        # 处理不同类型的点
        x = point.get('x', 0) if isinstance(point, dict) else getattr(point, 'x', 0)
        y = point.get('y', 0) if isinstance(point, dict) else getattr(point, 'y', 0)

        # 应用旋转变换
        x1 = x * cos_a - y * sin_a
        y1 = x * sin_a + y * cos_a

        # 保持与输入点相同的类型
        if isinstance(point, dict):
            rotated.append({'x': round(x1), 'y': round(y1)})

    # 复制原始多边形的属性
    for attr in dir(polygon):
        if not attr.startswith('__') and attr not in ('append', 'extend', 'x', 'y'):
            try:
                value = getattr(polygon, attr)
                if attr == 'children' and value:
                    # 递归旋转子多边形
                    rotated.children = [rotate_polygon(child, degrees) for child in value]
                else:
                    setattr(rotated, attr, value)
            except AttributeError:
                continue

    # 计算旋转后的边界框
    bounds = GeometryUtil.get_polygon_bounds(rotated)
    rotated.width = bounds['width']
    rotated.height = bounds['height']

    return rotated


class PlacementWorker:
    def __init__(self, bin_polygon: List[Dict], paths: List[Dict], ids: List[int],
                 rotations: List[float], config: Dict):
        """初始化放置工作器。

        Args:
            bin_polygon: 容器多边形
            paths: 要放置的路径列表
            ids: 路径ID列表
            rotations: 允许的旋转角度列表
            config: 配置参数
        """
        # 验证输入参数
        if not bin_polygon or not isinstance(bin_polygon, list):
            raise ValueError("无效的容器多边形")

        if not paths or not isinstance(paths, list):
            raise ValueError("无效的路径列表")

        if not ids or not isinstance(ids, list):
            raise ValueError("无效的ID列表")

        if not rotations or not isinstance(rotations, list):
            raise ValueError("无效的旋转角度列表")

        if not config or not isinstance(config, dict):
            raise ValueError("无效的配置参数")

        # 初始化属性
        self.bin_polygon = bin_polygon
        self.bin_id = -1  # 为容器多边形分配固定ID -1
        self.paths = paths
        self.ids = ids
        self.rotations = rotations
        self.config = config

        # 计算容器边界
        self.bin_bounds = GeometryUtil.get_polygon_bounds(bin_polygon)
        if not self.bin_bounds:
            raise ValueError("无法计算容器边界")

        # 计算容器面积
        self.bin_area = abs(GeometryUtil.polygon_area(bin_polygon))
        if self.bin_area <= 0:
            raise ValueError("容器面积无效")

        # 验证路径
        self.valid_paths = []
        self.total_area = 0
        for i, path in enumerate(paths):
            if not self._is_valid_polygon(path):
                print(f"路径 {i} 无效，跳过")
                continue

            area = abs(GeometryUtil.polygon_area(path))
            if area < 1e-6:
                print(f"路径 {i} 面积过小，跳过")
                continue

            self.valid_paths.append(path)
            self.total_area += area

        if not self.valid_paths:
            raise ValueError("没有有效的路径")

        print(f"初始化完成: {len(self.valid_paths)} 个有效路径，总面积={self.total_area}")

    def _get_reference_point(self, polygon: List[Dict]) -> Dict:
        """获取多边形的参考点（最左下角点）
        
        Args:
            polygon: 多边形点列表
            
        Returns:
            参考点坐标
        """
        if not polygon:
            return {'x': 0, 'y': 0}

        ref_point = {'x': polygon[0]['x'], 'y': polygon[0]['y']}
        for point in polygon:
            if point['x'] < ref_point['x'] or (point['x'] == ref_point['x'] and point['y'] < ref_point['y']):
                ref_point = {'x': point['x'], 'y': point['y']}
        return ref_point

    def place_paths(self):
        """放置路径，使用NFP进行优化放置"""
        print("place_paths: 开始放置路径...")

        paths = self.paths

        placed_paths = []
        unplaced = []
        fitness = 0
        max_area = 0

        # 1. 计算容器边界
        bin_bounds = GeometryUtil.get_polygon_bounds(self.bin_polygon)
        print(f"place_paths: 容器边界: {bin_bounds}")

        # 计算容器的左下角点（相对于原点）
        bin_min_x = bin_bounds['x']
        bin_min_y = bin_bounds['y']

        # 2. 旋转零件
        rotated = []
        for i, path in enumerate(paths):
            r = rotate_polygon(path, self.rotations[i])
            r.rotation= self.rotations[i]
            rotated.append(r)
        paths = rotated


        while paths:
            placed = []
            placements = []
            fitness += 1  # 每开启新的放置尝试增加1

            # 2. 尝试放置每个路径
            for i, path in enumerate(paths):
                # 获取路径的参考点
                ref_point = self._get_reference_point(path)

                # 获取路径ID
                path_id = getattr(path, 'id', i)

                # 获取与容器的内部NFP
                bin_nfp = GeometryUtil.no_fit_polygon(self.bin_polygon, path, inside=True)

                if not bin_nfp:
                    print(f"place_paths: 路径 {i} 无法放入容器")
                    continue

                # 如果是第一个放置的路径，放在容器左下角
                if not placed:
                    # 计算偏移量，确保路径被放置在容器内
                    dx = bin_min_x - ref_point['x']
                    dy = bin_min_y - ref_point['y']

                    position = {
                        'x': dx,
                        'y': dy,
                        'id': path_id,
                    }

                    # 验证位置是否有效
                    path_bounds = GeometryUtil.get_polygon_bounds(path)
                    if (dx + path_bounds['x'] < bin_min_x or
                            dy + path_bounds['y'] < bin_min_y or
                            dx + path_bounds['x'] + path_bounds['width'] > bin_min_x + bin_bounds['width'] or
                            dy + path_bounds['y'] + path_bounds['height'] > bin_min_y + bin_bounds['height']):
                        print(f"place_paths: 路径 {i} 无法放置在容器内")
                        continue

                    placements.append(position)
                    placed.append(path)
                    continue

                # 4. 计算与已放置路径的NFP
                combined_nfp = []

                for j, placed_path in enumerate(placed):
                    placed_id = getattr(placed_path, 'id', j)
                    nfp = GeometryUtil.no_fit_polygon(placed_path, path, inside=False)

                    if not nfp:
                        print(f"place_paths: 路径 {i} 与路径 {placed_id} 的NFP计算失败")
                        continue

                    # 移动NFP到已放置零件的位置
                    moved_nfp = []
                    for point in nfp:
                        moved_nfp.append({
                            'x': point['x'] + placements[j]['x'],
                            'y': point['y'] + placements[j]['y']
                        })
                    combined_nfp.append(moved_nfp)

                if not combined_nfp:
                    continue

                # 求combined_nfp图形轮廓的并集的外部轮廓与bin_nfp内部的交集
                final_nfp = GeometryUtil.polygon_intersection(combined_nfp, bin_nfp)

                # 排序，将final_nfp优先按y坐标从小到大排序,当y坐标相同时按x坐标从小到大排序
                final_nfp.sort(key=lambda point: (point['y'], point['x']))

                # 5. 在可行区域中寻找最佳位置
                min_width = None
                min_area = None
                best_position = None

                for k, point in enumerate(final_nfp):
                    # 计算放置后的所有点
                    all_points = []

                    # 添加已放置零件的点
                    for m, placed_item in enumerate(placed):
                        for n, p in enumerate(placed_item):
                            all_points.append({
                                'x': p['x'] + placements[m]['x'],
                                'y': p['y'] + placements[m]['y']
                            })

                    # 计算当前位置
                    shift = {
                        'x': point['x'] - ref_point['x'],
                        'y': point['y'] - ref_point['y'],
                        'id': path_id,
                        'rotation': getattr(path, 'rotation', 0)
                    }

                    # 添加当前零件的点
                    for p in path:
                        all_points.append({
                            'x': p['x'] + shift['x'],
                            'y': p['y'] + shift['y']
                        })

                    # 计算边界框
                    bounds = GeometryUtil.get_polygon_bounds(all_points)

                    # 检查是否在容器内
                    if (bounds['x'] < bin_min_x or bounds['y'] < bin_min_y or
                            bounds['x'] + bounds['width'] > bin_min_x + bin_bounds['width'] or
                            bounds['y'] + bounds['height'] > bin_min_y + bin_bounds['height']):
                        continue

                    # 计算面积（权重宽度更大，以帮助压缩重力方向）
                    area = bounds['height']

                    if min_area is None or area < min_area or area == min_area and bounds['width'] < min_width:
                        max_area = bounds['height']*bounds['width']
                        min_width = bounds['width']
                        min_area = area
                        best_position = shift

                if best_position:
                    placed.append(path)
                    placements.append(best_position)

            # 6. 更新未放置的路径
            for path in paths:
                if path not in placed:
                    unplaced.append(path)

            paths = unplaced
            unplaced = []

            # 7. 构建当前批次的放置结果
            for i, placement in enumerate(placements):
                # 找到对应的原始路径
                original_path = placed[i]

                # 创建放置后的路径
                CustomList = type('CustomList', (list,), {})
                placed_path = CustomList()
                for point in original_path:
                    placed_path.append({
                        'x': point['x'] + placement['x'],
                        'y': point['y'] + placement['y']
                    })
                placed_path.id = original_path.id
                placed_path.rotation = original_path.rotation
                placed_paths.append(placed_path)
        print(f"place_paths: 完成，放置 {len(placed_paths)} 个路径，未放置 {len(unplaced)} 个路径")
        return placed_paths, unplaced, max_area

    def _move_path(self, path: List[Dict], position: Dict) -> List[Dict]:
        """移动路径到指定位置
        
        Args:
            path: 要移动的路径
            position: 目标位置
            
        Returns:
            移动后的路径
        """
        moved_path = []
        for point in path:
            moved_path.append({
                'x': point['x'] + position['x'],
                'y': point['y'] + position['y']
            })
        return moved_path

    def _is_valid_polygon(self, polygon: List[Dict]) -> bool:
        """检查多边形是否有效
        
        Args:
            polygon: 多边形点列表
            
        Returns:
            多边形是否有效
        """
        try:
            if not polygon or len(polygon) < 3:
                print("_is_valid_polygon: 多边形点数不足")
                return False

            # 检查点格式和类型
            valid_points = []
            for point in polygon:
                # 处理元组格式
                if isinstance(point, (list, tuple)):
                    if len(point) < 2:
                        print("_is_valid_polygon: 点格式无效 - 元组长度不足")
                        return False
                    try:
                        x = float(point[0])
                        y = float(point[1])
                        valid_points.append({'x': x, 'y': y})
                    except (ValueError, TypeError):
                        print("_is_valid_polygon: 点坐标转换失败")
                        return False
                # 处理字典格式
                elif isinstance(point, dict):
                    if 'x' not in point or 'y' not in point:
                        print("_is_valid_polygon: 点格式无效 - 缺少x或y坐标")
                        return False
                    try:
                        x = float(point['x'])
                        y = float(point['y'])
                        valid_points.append({'x': x, 'y': y})
                    except (ValueError, TypeError):
                        print("_is_valid_polygon: 点坐标转换失败")
                        return False
                else:
                    print(f"_is_valid_polygon: 不支持的点格式: {type(point)}")
                    return False

                # 检查坐标值是否有效
                if math.isnan(x) or math.isinf(x) or math.isnan(y) or math.isinf(y):
                    print("_is_valid_polygon: 坐标值无效")
                    return False

                if abs(x) > 1e12 or abs(y) > 1e12:
                    print("_is_valid_polygon: 坐标值过大")
                    return False

            # 检查是否有重复点
            for i in range(len(valid_points) - 1):
                if GeometryUtil.almost_equal(valid_points[i]['x'], valid_points[i + 1]['x'], 0.1) and \
                        GeometryUtil.almost_equal(valid_points[i]['y'], valid_points[i + 1]['y'], 0.1):
                    print("_is_valid_polygon: 发现重复点")
                    return False

            # 计算面积
            area = abs(self._calculate_area(valid_points))
            if area < 1e-4:  # 增加面积阈值
                print("_is_valid_polygon: 面积过小")
                return False

            return True

        except Exception as e:
            print(f"_is_valid_polygon: 验证过程中发生错误: {e}")
            return False

    def _calculate_area(self, polygon: List[Dict]) -> float:
        """计算多边形面积"""
        area = 0
        j = len(polygon) - 1

        for i in range(len(polygon)):
            p1 = polygon[j]
            p2 = polygon[i]
            p1_x = p1['x']
            p1_y = p1['y']
            p2_x = p2['x']
            p2_y = p2['y']

            area += (p1_x + p2_x) * (p1_y - p2_y)
            j = i

        return area / 2

    # 在geometry_util.py中有
    # def calculate_nfp_cache(self, bin_polygon, paths):
    #     """计算NFP缓存
    #
    #     Args:
    #         bin_polygon: 容器多边形
    #         paths: 路径列表
    #
    #     Returns:
    #         NFP缓存字典
    #     """
    #     print("calculate_nfp_cache: 开始计算...")
    #
    #     # 验证容器多边形
    #     if not self._is_valid_polygon(bin_polygon):
    #         print("calculate_nfp_cache: 容器多边形无效")
    #         return {}
    #
    #     bin_area = abs(GeometryUtil.polygon_area(bin_polygon))
    #     print(f"calculate_nfp_cache: 容器多边形有效，面积={bin_area}")
    #
    #     # 验证路径
    #     valid_paths = []
    #     total_area = 0
    #     for path in paths:
    #         if not self._is_valid_polygon(path):
    #             print(f"calculate_nfp_cache: 路径 {len(valid_paths)} 无效，跳过")
    #             continue
    #
    #         area = abs(GeometryUtil.polygon_area(path))
    #         if area < 1e-6:
    #             print(f"calculate_nfp_cache: 路径 {len(valid_paths)} 面积过小，跳过")
    #             continue
    #
    #         print(f"calculate_nfp_cache: 路径 {len(valid_paths)} 有效，面积={area}")
    #         valid_paths.append(path)
    #         total_area += area
    #
    #     print(f"calculate_nfp_cache: 有效路径数量={len(valid_paths)}，总面积={total_area}")
    #
    #     # 初始化NFP缓存
    #     nfp_cache = {}
    #
    #     # 计算每个路径与容器的NFP
    #     for i, path in enumerate(valid_paths):
    #         print(f"calculate_nfp_cache: 处理路径 {i}")
    #
    #         # 计算路径与容器的NFP
    #         nfp = GeometryUtil.no_fit_polygon(bin_polygon, path, inside=True)
    #         if nfp:
    #             key = f"bin,{i}"
    #             nfp_cache[key] = nfp
    #             print(f"calculate_nfp_cache: 成功计算路径 {i} 与容器的NFP")
    #         else:
    #             print(f"calculate_nfp_cache: 路径 {i} 与容器的NFP计算失败")
    #             continue
    #
    #         # 计算路径与其他路径的NFP
    #         for j in range(i + 1, len(valid_paths)):
    #             print(f"calculate_nfp_cache: 计算路径 {i} 和 {j} 之间的NFP")
    #
    #             # 尝试计算NFP
    #             nfp = GeometryUtil.no_fit_polygon(path, valid_paths[j])
    #             if nfp:
    #                 key = f"{i},{j}"
    #                 nfp_cache[key] = nfp
    #                 print(f"calculate_nfp_cache: 成功计算路径 {i} 和 {j} 之间的NFP")
    #             else:
    #                 print(f"calculate_nfp_cache: 路径 {i} 和 {j} 之间的NFP计算失败")
    #
    #                 # 尝试反向计算
    #                 nfp = GeometryUtil.no_fit_polygon(valid_paths[j], path)
    #                 if nfp:
    #                     key = f"{j},{i}"
    #                     nfp_cache[key] = nfp
    #                     print(f"calculate_nfp_cache: 成功计算路径 {j} 和 {i} 之间的NFP（反向）")
    #                 else:
    #                     print(f"calculate_nfp_cache: 路径 {j} 和 {i} 之间的NFP计算失败（反向）")
    #
    #     print(f"calculate_nfp_cache: 完成，共计算 {len(nfp_cache)} 个NFP")
    #     return nfp_cache
