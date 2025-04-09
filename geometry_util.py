# geometry_util.py

import math
from typing import List, Dict, Any, Optional, Union, Tuple
import pyclipper

class GeometryUtil:
    """几何工具类"""
    
    @staticmethod
    def almost_equal(a: float, b: float, tolerance: float = 0.001) -> bool:
        """判断两个浮点数是否近似相等"""
        return abs(a - b) < tolerance

    @staticmethod
    def within_distance(p1: Dict, p2: Dict, distance: float) -> bool:
        """判断两点是否在给定距离内"""
        dx = p1['x'] - p2['x']
        dy = p1['y'] - p2['y']
        return dx * dx + dy * dy <= distance * distance

    @staticmethod
    def degrees_to_radians(angle: float) -> float:
        """角度转弧度"""
        return angle * math.pi / 180

    @staticmethod
    def radians_to_degrees(angle: float) -> float:
        """弧度转角度"""
        return angle * 180 / math.pi

    @staticmethod
    def normalize_vector(v: Dict) -> Dict:
        """向量归一化"""
        length = math.sqrt(v['x'] * v['x'] + v['y'] * v['y'])
        if length == 0:
            return {'x': 0, 'y': 0}
        return {'x': v['x'] / length, 'y': v['y'] / length}

    @staticmethod
    def on_segment(A: Dict, B: Dict, p: Dict) -> bool:
        """判断点p是否在线段AB上"""
        # 检查点p是否在线段AB的边界框内
        if (p['x'] <= max(A['x'], B['x']) and p['x'] >= min(A['x'], B['x']) and
            p['y'] <= max(A['y'], B['y']) and p['y'] >= min(A['y'], B['y'])):
            
            # 检查点p是否在线段AB上
            cross_product = ((p['y'] - A['y']) * (B['x'] - A['x']) -
                           (p['x'] - A['x']) * (B['y'] - A['y']))
            
            return abs(cross_product) < 1e-10
            
        return False

    @staticmethod
    def line_intersect(A: Dict, B: Dict, E: Dict, F: Dict, infinite: bool = False) -> Optional[Dict]:
        """
        计算两条线段的交点
        infinite: 如果为True，则将线段视为无限延长的直线
        """
        # 线段AB的方向向量
        dxAB = B['x'] - A['x']
        dyAB = B['y'] - A['y']
        
        # 线段EF的方向向量
        dxEF = F['x'] - E['x']
        dyEF = F['y'] - E['y']
        
        # 计算行列式
        det = dxAB * dyEF - dyAB * dxEF
        
        if abs(det) < 1e-10:  # 平行或重合
            return None
            
        # 计算参数t和s
        t = ((E['x'] - A['x']) * dyEF - (E['y'] - A['y']) * dxEF) / det
        s = ((E['x'] - A['x']) * dyAB - (E['y'] - A['y']) * dxAB) / det
        
        # 如果不是无限延长线，检查参数是否在[0,1]范围内
        if not infinite and (t < 0 or t > 1 or s < 0 or s > 1):
            return None
            
        # 计算交点
        return {
            'x': A['x'] + t * dxAB,
            'y': A['y'] + t * dyAB
        }

    @staticmethod
    def polygon_area(polygon: List[Dict]) -> float:
        """计算多边形面积（正值表示逆时针，负值表示顺时针）"""
        area = 0
        j = len(polygon) - 1
        
        for i in range(len(polygon)):
            area += (polygon[j]['x'] + polygon[i]['x']) * (polygon[j]['y'] - polygon[i]['y'])
            j = i
            
        return area / 2

    @staticmethod
    def get_polygon_bounds(polygon: List[Dict]) -> Dict:
        """获取多边形的边界框"""
        if not polygon:
            return {'x': 0, 'y': 0, 'width': 0, 'height': 0}
            
        minx = maxx = polygon[0]['x']
        miny = maxy = polygon[0]['y']
        
        for p in polygon[1:]:
            minx = min(minx, p['x'])
            miny = min(miny, p['y'])
            maxx = max(maxx, p['x'])
            maxy = max(maxy, p['y'])
            
        return {
            'x': minx,
            'y': miny,
            'width': maxx - minx,
            'height': maxy - miny
        }

    @staticmethod
    def rotate_polygon(polygon: List[Dict], angle: float) -> List[Dict]:
        """旋转多边形"""
        rad = GeometryUtil.degrees_to_radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        rotated = []
        for p in polygon:
            rotated.append({
                'x': p['x'] * cos_a - p['y'] * sin_a,
                'y': p['x'] * sin_a + p['y'] * cos_a
            })
            
        # 确保返回的是一个普通列表，而不是继承了输入多边形的类型
        return list(rotated)

    @staticmethod
    def no_fit_polygon(polygon_a, polygon_b, inside=False):
        """计算两个多边形之间的NFP（No Fit Polygon）
        Args:
            polygon_a: 第一个多边形
            polygon_b: 第二个多边形
            inside: 是否计算内部NFP
        Returns:
            NFP点列表或None（如果计算失败）
        """
        # 输入验证
        if not polygon_a or not polygon_b:
            print("no_fit_polygon: 输入多边形无效")
            return None
        
        # 深拷贝输入多边形以避免修改原始数据
        polygon_a = list(polygon_a)
        polygon_b = list(polygon_b)
        
        # 确保输入多边形闭合
        if len(polygon_a) >= 2 and (not GeometryUtil.almost_equal(polygon_a[0]['x'], polygon_a[-1]['x'], 0.1) or 
                                   not GeometryUtil.almost_equal(polygon_a[0]['y'], polygon_a[-1]['y'], 0.1)):
            polygon_a.append(dict(polygon_a[0]))
            print("no_fit_polygon: 自动闭合多边形A")
            
        if len(polygon_b) >= 2 and (not GeometryUtil.almost_equal(polygon_b[0]['x'], polygon_b[-1]['x'], 0.1) or 
                                   not GeometryUtil.almost_equal(polygon_b[0]['y'], polygon_b[-1]['y'], 0.1)):
            polygon_b.append(dict(polygon_b[0]))
            print("no_fit_polygon: 自动闭合多边形B")
        
        # 清理输入多边形
        polygon_a = GeometryUtil.clean_polygon(polygon_a)
        polygon_b = GeometryUtil.clean_polygon(polygon_b)
        
        if not polygon_a or not polygon_b:
            print("no_fit_polygon: 清理后的多边形无效")
            return None
        
        # 计算多边形面积
        area_a = abs(GeometryUtil.polygon_area(polygon_a))
        area_b = abs(GeometryUtil.polygon_area(polygon_b))
        
        if area_a < 1e-6 or area_b < 1e-6:
            print("no_fit_polygon: 多边形面积过小")
            return None
            
        # 确保多边形方向一致（逆时针）
        if GeometryUtil.polygon_area(polygon_a) < 0:
            polygon_a.reverse()
        if GeometryUtil.polygon_area(polygon_b) < 0:
            polygon_b.reverse()
        
        if inside:
            # 获取容器的边界
            min_x = min(p['x'] for p in polygon_a)
            min_y = min(p['y'] for p in polygon_a)
            max_x = max(p['x'] for p in polygon_a)
            max_y = max(p['y'] for p in polygon_a)
            
            # 获取移动多边形的边界
            b_min_x = min(p['x'] for p in polygon_b)
            b_min_y = min(p['y'] for p in polygon_b)
            b_width = max(p['x'] for p in polygon_b) - b_min_x
            b_height = max(p['y'] for p in polygon_b) - b_min_y
            
            # 构建NFP点
            nfp_points = []
            step = min(b_width, b_height) / 20  # 减小步长以提高精度
            
            # 确保步长不会太小
            if step < 0.01:  # 减小最小步长
                step = 0.01
                
            # 在容器内部生成网格点
            for x in range(int((max_x - min_x - b_width) / step) + 1):
                for y in range(int((max_y - min_y - b_height) / step) + 1):
                    point = {
                        'x': min_x + x * step,
                        'y': min_y + y * step
                    }
                    
                    # 检查移动后的多边形B是否完全在容器内部
                    valid = True
                    for b_point in polygon_b:
                        test_point = {
                            'x': point['x'] + (b_point['x'] - b_min_x),
                            'y': point['y'] + (b_point['y'] - b_min_y)
                        }
                        if not GeometryUtil.point_in_polygon(test_point, polygon_a):
                            valid = False
                            break
                    
                    if valid:
                        nfp_points.append(point)
            
            if len(nfp_points) < 3:
                print("no_fit_polygon: NFP点数不足")
                return None
            
            # 对点进行凸包计算以形成有效的多边形
            nfp_points = GeometryUtil.convex_hull(nfp_points)
            
            if not nfp_points:
                print("no_fit_polygon: 无法构建有效的NFP")
                return None
            
            # 确保NFP闭合
            if len(nfp_points) >= 2 and (not GeometryUtil.almost_equal(nfp_points[0]['x'], nfp_points[-1]['x'], 0.1) or 
                                       not GeometryUtil.almost_equal(nfp_points[0]['y'], nfp_points[-1]['y'], 0.1)):
                nfp_points.append(dict(nfp_points[0]))
                print("no_fit_polygon: 自动闭合NFP")
            
            # 验证NFP面积
            nfp_area = abs(GeometryUtil.polygon_area(nfp_points))
            if nfp_area < 1e-6:
                print("no_fit_polygon: NFP面积过小")
                return None
            
            return nfp_points
        
        # 外部NFP计算
        try:
            # 使用pyclipper计算NFP
            pc = pyclipper.Pyclipper()
            
            # 转换为Clipper格式
            scale = 1000000
            a_path = [(int(p['x'] * scale), int(p['y'] * scale)) for p in polygon_a]
            b_path = [(int(p['x'] * scale), int(p['y'] * scale)) for p in polygon_b]
            
            # 添加路径
            pc.AddPath(a_path, pyclipper.PT_SUBJECT, True)
            pc.AddPath(b_path, pyclipper.PT_CLIP, True)
            
            # 执行Minkowski和
            solution = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
            
            if not solution:
                print("no_fit_polygon: Clipper计算失败")
                return None
            
            # 转换回原始坐标
            nfp = []
            for path in solution:
                points = []
                for point in path:
                    points.append({
                        'x': point[0] / scale,
                        'y': point[1] / scale
                    })
                if len(points) >= 3:
                    # 确保闭合
                    if not (GeometryUtil.almost_equal(points[0]['x'], points[-1]['x'], 0.1) and 
                           GeometryUtil.almost_equal(points[0]['y'], points[-1]['y'], 0.1)):
                        points.append(dict(points[0]))
                    nfp.extend(points)
            
            if not nfp:
                print("no_fit_polygon: 未生成有效的NFP")
                return None
            
            return nfp
            
        except Exception as e:
            print(f"no_fit_polygon: NFP计算出错: {e}")
            return None

    @staticmethod
    def convex_hull(points):
        """计算点集的凸包
        使用Graham扫描法
        """
        if len(points) < 3:
            return points
            
        # 找到最低点
        bottom_point = min(points, key=lambda p: (p['y'], p['x']))
        
        # 计算极角并排序
        def polar_angle(p):
            dx = p['x'] - bottom_point['x']
            dy = p['y'] - bottom_point['y']
            if dx == 0 and dy == 0:
                return float('-inf')
            return math.atan2(dy, dx)
            
        sorted_points = sorted(points, key=polar_angle)
        
        # Graham扫描
        stack = [sorted_points[0], sorted_points[1]]
        
        for i in range(2, len(sorted_points)):
            while len(stack) > 1:
                p1 = stack[-2]
                p2 = stack[-1]
                p3 = sorted_points[i]
                
                # 计算叉积
                cross_product = (p2['x'] - p1['x']) * (p3['y'] - p1['y']) - \
                              (p2['y'] - p1['y']) * (p3['x'] - p1['x'])
                              
                if cross_product > 0:
                    break
                stack.pop()
                
            stack.append(sorted_points[i])
            
        # 闭合多边形
        if stack[0] != stack[-1]:
            stack.append(stack[0])
            
        return stack

    @staticmethod
    def point_in_polygon(point: Dict, polygon: List[Dict]) -> bool:
        """判断点是否在多边形内部"""
        if not polygon:
            return False
            
        inside = False
        j = len(polygon) - 1
        
        for i in range(len(polygon)):
            if (((polygon[i]['y'] > point['y']) != (polygon[j]['y'] > point['y'])) and
                (point['x'] < (polygon[j]['x'] - polygon[i]['x']) * 
                 (point['y'] - polygon[i]['y']) / (polygon[j]['y'] - polygon[i]['y']) + 
                 polygon[i]['x'])):
                inside = not inside
            j = i
            
        return inside

    @staticmethod
    def clean_polygon(polygon: List[Dict]) -> List[Dict]:
        """清理多边形路径，确保其有效性
        
        Args:
            polygon: 多边形点列表
            
        Returns:
            清理后的多边形点列表
        """
        if not polygon or len(polygon) < 3:
            print("clean_polygon: 多边形无效，点数不足")
            return []
            
        # 验证所有点是否有效
        valid_points = []
        for point in polygon:
            if isinstance(point, dict) and 'x' in point and 'y' in point:
                # 检查坐标是否为有效数字
                try:
                    x = float(point['x'])
                    y = float(point['y'])
                    if not (math.isnan(x) or math.isinf(x) or math.isnan(y) or math.isinf(y)):
                        valid_points.append(point)
                except (ValueError, TypeError):
                    print("clean_polygon: 发现无效坐标值，跳过")
            else:
                print("clean_polygon: 发现无效点格式，跳过")
                
        if len(valid_points) < 3:
            print("clean_polygon: 有效点数不足")
            return []
            
        # 移除重复点
        cleaned = []
        for point in valid_points:
            if not cleaned or not GeometryUtil.almost_equal(point['x'], cleaned[-1]['x']) or \
               not GeometryUtil.almost_equal(point['y'], cleaned[-1]['y']):
                cleaned.append(point)
                
        if len(cleaned) < 3:
            print("clean_polygon: 清理后点数不足")
            return []
            
        # 确保多边形闭合
        if cleaned and (not GeometryUtil.almost_equal(cleaned[0]['x'], cleaned[-1]['x']) or \
                       not GeometryUtil.almost_equal(cleaned[0]['y'], cleaned[-1]['y'])):
            cleaned.append(cleaned[0])
            
        # 计算多边形面积
        area = GeometryUtil.polygon_area(cleaned)
        if abs(area) < 1e-6:  # 使用更小的面积阈值
            print("clean_polygon: 多边形面积过小")
            return []
            
        return cleaned

    @staticmethod
    def has_self_intersections(polygon: List[Dict]) -> bool:
        """检查多边形是否自相交"""
        n = len(polygon)
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                if GeometryUtil.line_intersect(polygon[i], polygon[i + 1], 
                                             polygon[j], polygon[(j + 1) % n]):
                    return True
        return False

    @staticmethod
    def fix_self_intersections(polygon: List[Dict]) -> List[Dict]:
        """尝试修复自相交多边形"""
        try:
            # 使用简单的修复方法：找到自相交点，分割多边形
            n = len(polygon)
            for i in range(n):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    intersection = GeometryUtil.line_intersect(
                        polygon[i], polygon[i + 1],
                        polygon[j], polygon[(j + 1) % n]
                    )
                    if intersection:
                        # 分割多边形
                        part1 = polygon[:i + 1] + [intersection] + polygon[j + 1:]
                        part2 = [intersection] + polygon[i + 1:j + 1]
                        
                        # 选择面积较大的部分
                        area1 = abs(GeometryUtil.polygon_area(part1))
                        area2 = abs(GeometryUtil.polygon_area(part2))
                        return part1 if area1 > area2 else part2
            return polygon
        except Exception as e:
            print(f"fix_self_intersections: 修复失败: {e}")
            return None

    @staticmethod
    def is_valid_polygon(polygon: List[Dict]) -> bool:
        """检查多边形是否有效
        
        Args:
            polygon: 多边形点列表
            
        Returns:
            多边形是否有效
        """
        try:
            if not polygon or len(polygon) < 3:
                print("is_valid_polygon: 多边形无效，点数不足")
                return False
                
            # 检查是否有足够的点
            if len(polygon) < 3:
                print("is_valid_polygon: 点数不足")
                return False
                
            # 检查点格式和类型
            for point in polygon:
                if not isinstance(point, dict) or 'x' not in point or 'y' not in point:
                    print("is_valid_polygon: 点格式无效")
                    return False
                    
                try:
                    x = float(point['x'])
                    y = float(point['y'])
                    if math.isnan(x) or math.isinf(x) or math.isnan(y) or math.isinf(y):
                        print("is_valid_polygon: 坐标值无效")
                        return False
                        
                    if abs(x) > 1e12 or abs(y) > 1e12:
                        print("is_valid_polygon: 坐标值过大")
                        return False
                except (ValueError, TypeError):
                    print("is_valid_polygon: 坐标转换失败")
                    return False
                    
            # 检查是否闭合
            if not GeometryUtil.almost_equal(polygon[0]['x'], polygon[-1]['x']) or \
               not GeometryUtil.almost_equal(polygon[0]['y'], polygon[-1]['y']):
                print("is_valid_polygon: 多边形未闭合")
                return False
                
            # 检查是否有重复点
            for i in range(len(polygon)-1):
                if GeometryUtil.almost_equal(polygon[i]['x'], polygon[i+1]['x']) and \
                   GeometryUtil.almost_equal(polygon[i]['y'], polygon[i+1]['y']):
                    print("is_valid_polygon: 发现重复点")
                    return False
                    
            # 检查面积
            area = abs(GeometryUtil.polygon_area(polygon))
            if area < 1e-4:  # 增加面积阈值
                print("is_valid_polygon: 面积过小")
                return False
                
            # 检查自相交
            if GeometryUtil.has_self_intersections(polygon):
                print("is_valid_polygon: 多边形自相交")
                return False
                
            return True
            
        except Exception as e:
            print(f"is_valid_polygon: 验证过程中发生错误: {e}")
            return False

    @staticmethod
    def polygons_intersect(polygon_a: List[Dict], polygon_b: List[Dict]) -> bool:
        """检查两个多边形是否相交
        
        Args:
            polygon_a: 第一个多边形
            polygon_b: 第二个多边形
            
        Returns:
            如果多边形相交返回True，否则返回False
        """
        # 检查每条边是否与另一个多边形的任意边相交
        for i in range(len(polygon_a)):
            i_next = (i + 1) % len(polygon_a)
            edge_a = (polygon_a[i], polygon_a[i_next])
            
            for j in range(len(polygon_b)):
                j_next = (j + 1) % len(polygon_b)
                edge_b = (polygon_b[j], polygon_b[j_next])
                
                # 检查边是否相交
                if GeometryUtil.edges_intersect((edge_a[0], edge_a[1]), (edge_b[0], edge_b[1])):
                    return True
                    
        # 检查一个多边形是否完全包含另一个多边形
        if GeometryUtil.point_in_polygon(polygon_a[0], polygon_b) or \
           GeometryUtil.point_in_polygon(polygon_b[0], polygon_a):
            return True
            
        return False
        
    @staticmethod
    def edges_intersect(edge_a: Tuple[Dict, Dict], edge_b: Tuple[Dict, Dict]) -> bool:
        """检查两条边是否相交
        
        Args:
            edge_a: 第一条边的两个端点
            edge_b: 第二条边的两个端点
            
        Returns:
            如果边相交返回True，否则返回False
        """
        # 计算叉积
        def cross_product(p1: Dict, p2: Dict, p3: Dict) -> float:
            return (p2['x'] - p1['x']) * (p3['y'] - p1['y']) - \
                   (p2['y'] - p1['y']) * (p3['x'] - p1['x'])
                   
        # 检查点是否在线段上
        def on_segment(p: Dict, q: Dict, r: Dict) -> bool:
            return q['x'] <= max(p['x'], r['x']) and q['x'] >= min(p['x'], r['x']) and \
                   q['y'] <= max(p['y'], r['y']) and q['y'] >= min(p['y'], r['y'])
                   
        p1, q1 = edge_a
        p2, q2 = edge_b
        
        o1 = cross_product(p1, q1, p2)
        o2 = cross_product(p1, q1, q2)
        o3 = cross_product(p2, q2, p1)
        o4 = cross_product(p2, q2, q1)
        
        # 一般情况下的相交
        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
            
        # 特殊情况：边重合或端点在另一条边上
        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        if o4 == 0 and on_segment(p2, q1, q2):
            return True
            
        return False

    class QuadraticBezier:
        """二次贝塞尔曲线工具"""
        
        @staticmethod
        def linearize(p0: Dict, p2: Dict, p1: Dict, tolerance: float) -> List[Dict]:
            """将二次贝塞尔曲线线性化"""
            points = [p0]
            GeometryUtil.QuadraticBezier._recursive_linearize(
                p0, p2, p1, tolerance, points
            )
            points.append(p2)
            return points

        @staticmethod
        def _recursive_linearize(p0: Dict, p2: Dict, p1: Dict, 
                               tolerance: float, points: List[Dict]):
            """递归线性化二次贝塞尔曲线"""
            # 计算曲线中点
            mid = {
                'x': (p0['x'] + 2 * p1['x'] + p2['x']) / 4,
                'y': (p0['y'] + 2 * p1['y'] + p2['y']) / 4
            }
            
            # 计算直线中点
            line_mid = {
                'x': (p0['x'] + p2['x']) / 2,
                'y': (p0['y'] + p2['y']) / 2
            }
            
            # 计算误差
            dx = mid['x'] - line_mid['x']
            dy = mid['y'] - line_mid['y']
            error = dx * dx + dy * dy
            
            if error <= tolerance:
                points.append(mid)
            else:
                # 分割曲线并递归
                p0_1 = {
                    'x': (p0['x'] + p1['x']) / 2,
                    'y': (p0['y'] + p1['y']) / 2
                }
                p1_2 = {
                    'x': (p1['x'] + p2['x']) / 2,
                    'y': (p1['y'] + p2['y']) / 2
                }
                
                GeometryUtil.QuadraticBezier._recursive_linearize(
                    p0, mid, p0_1, tolerance, points
                )
                GeometryUtil.QuadraticBezier._recursive_linearize(
                    mid, p2, p1_2, tolerance, points
                )

    class CubicBezier:
        """三次贝塞尔曲线工具"""
        
        @staticmethod
        def linearize(p0: Dict, p3: Dict, p1: Dict, p2: Dict, 
                     tolerance: float) -> List[Dict]:
            """将三次贝塞尔曲线线性化"""
            points = [p0]
            GeometryUtil.CubicBezier._recursive_linearize(
                p0, p3, p1, p2, tolerance, points
            )
            points.append(p3)
            return points

        @staticmethod
        def _recursive_linearize(p0: Dict, p3: Dict, p1: Dict, p2: Dict, 
                               tolerance: float, points: List[Dict]):
            """递归线性化三次贝塞尔曲线"""
            # 计算曲线中点
            mid = {
                'x': (p0['x'] + 3 * (p1['x'] + p2['x']) + p3['x']) / 8,
                'y': (p0['y'] + 3 * (p1['y'] + p2['y']) + p3['y']) / 8
            }
            
            # 计算直线中点
            line_mid = {
                'x': (p0['x'] + p3['x']) / 2,
                'y': (p0['y'] + p3['y']) / 2
            }
            
            # 计算误差
            dx = mid['x'] - line_mid['x']
            dy = mid['y'] - line_mid['y']
            error = dx * dx + dy * dy
            
            if error <= tolerance:
                points.append(mid)
            else:
                # 分割曲线并递归
                p0_1 = {
                    'x': (p0['x'] + p1['x']) / 2,
                    'y': (p0['y'] + p1['y']) / 2
                }
                p1_2 = {
                    'x': (p1['x'] + p2['x']) / 2,
                    'y': (p1['y'] + p2['y']) / 2
                }
                p2_3 = {
                    'x': (p2['x'] + p3['x']) / 2,
                    'y': (p2['y'] + p3['y']) / 2
                }
                p01_12 = {
                    'x': (p0_1['x'] + p1_2['x']) / 2,
                    'y': (p0_1['y'] + p1_2['y']) / 2
                }
                p12_23 = {
                    'x': (p1_2['x'] + p2_3['x']) / 2,
                    'y': (p1_2['y'] + p2_3['y']) / 2
                }
                
                GeometryUtil.CubicBezier._recursive_linearize(
                    p0, mid, p0_1, p01_12, tolerance, points
                )
                GeometryUtil.CubicBezier._recursive_linearize(
                    mid, p3, p12_23, p2_3, tolerance, points
                )

    class Arc:
        """圆弧工具"""
        
        @staticmethod
        def linearize(start: Dict, end: Dict, rx: float, ry: float, 
                     angle: float, large_arc: bool, sweep: bool, 
                     tolerance: float) -> List[Dict]:
            """将圆弧线性化"""
            # 将角度转换为弧度
            angle_rad = GeometryUtil.degrees_to_radians(angle)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)
            
            # 计算中心点
            dx = (start['x'] - end['x']) / 2
            dy = (start['y'] - end['y']) / 2
            
            x1 = cos_angle * dx + sin_angle * dy
            y1 = -sin_angle * dx + cos_angle * dy
            
            # 确保半径足够大
            rx = abs(rx)
            ry = abs(ry)
            x1_sq = x1 * x1
            y1_sq = y1 * y1
            rx_sq = rx * rx
            ry_sq = ry * ry
            
            lambda_sq = x1_sq / rx_sq + y1_sq / ry_sq
            if lambda_sq > 1:
                lambda_sqrt = math.sqrt(lambda_sq)
                rx *= lambda_sqrt
                ry *= lambda_sqrt
                rx_sq = rx * rx
                ry_sq = ry * ry
                
            # 计算中心点
            c_sign = -1 if large_arc == sweep else 1
            c_term = c_sign * math.sqrt(
                max(0, (rx_sq * ry_sq - rx_sq * y1_sq - ry_sq * x1_sq) / 
                    (rx_sq * y1_sq + ry_sq * x1_sq))
            )
            
            cx1 = c_term * rx * y1 / ry
            cy1 = -c_term * ry * x1 / rx
            
            cx = cos_angle * cx1 - sin_angle * cy1 + (start['x'] + end['x']) / 2
            cy = sin_angle * cx1 + cos_angle * cy1 + (start['y'] + end['y']) / 2
            
            # 计算角度
            ux = (x1 - cx1) / rx
            uy = (y1 - cy1) / ry
            vx = (-x1 - cx1) / rx
            vy = (-y1 - cy1) / ry
            
            start_angle = math.atan2(uy, ux)
            delta_angle = math.atan2(vy * ux - vx * uy, vx * ux + vy * uy)
            
            if not sweep and delta_angle > 0:
                delta_angle -= 2 * math.pi
            elif sweep and delta_angle < 0:
                delta_angle += 2 * math.pi
                
            # 线性化
            num_segments = max(
                math.ceil(abs(delta_angle) / math.acos(1 - tolerance / max(rx, ry))),
                4
            )
            
            points = [start]
            for i in range(1, num_segments):
                t = i / num_segments
                angle = start_angle + t * delta_angle
                
                x = cos_angle * rx * math.cos(angle) - sin_angle * ry * math.sin(angle) + cx
                y = sin_angle * rx * math.cos(angle) + cos_angle * ry * math.sin(angle) + cy
                
                points.append({'x': x, 'y': y})
                
            points.append(end)
            return points