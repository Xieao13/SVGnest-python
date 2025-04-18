# svg_parser.py

import math
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from xml.dom import minidom
import pyclipper
import numpy as np
from svg.path import parse_path, Line, CubicBezier, QuadraticBezier, Arc, Path
from svgpathtools import parse_path as parse_path_complex
from matrix import Matrix
from geometry_util import GeometryUtil


class SvgParser:
    def __init__(self):
        """初始化SVG解析器"""
        self.svg = None
        self.svg_root = None
        self.style = None
        self.allowed_elements = ['svg', 'circle', 'ellipse', 'path', 'polygon', 'polyline', 'rect', 'line']
        self.conf = {
            'tolerance': 2,  # bezier->line段转换的最大边界
            'toleranceSvg': 0.005  # 浏览器SVG单位处理的容差
        }

    def clean_input(self):
        """清理并处理SVG输入"""
        if not self.svg_root:
            return None

        # 应用任何变换
        self.apply_transform(self.svg_root)

        # 移除g元素并将所有元素提升到顶层
        self.flatten(self.svg_root)

        # 移除非轮廓元素如文本
        self.filter(self.allowed_elements)

        # 分割复合路径为单独的路径元素
        self.recurse(self.svg_root, self.split_path)

        return self.svg_root

    def apply_transform(self, element):
        """应用SVG变换"""
        if not element or element.nodeType != element.ELEMENT_NODE:
            return

        # 获取变换矩阵
        transform = element.getAttribute('transform')
        if transform:
            matrix = Matrix()

            # 解析变换字符串
            transforms = transform.lower().split(')')
            for t in transforms:
                t = t.strip()
                if not t:
                    continue

                # 提取变换类型和参数
                type_start = t.find('(')
                if type_start == -1:
                    continue

                transform_type = t[:type_start].strip()
                args = [float(x) for x in t[type_start + 1:].split(',')]

                # 应用相应的变换
                if transform_type == 'translate':
                    if len(args) == 1:
                        matrix.translate(args[0], 0)
                    else:
                        matrix.translate(args[0], args[1])
                elif transform_type == 'rotate':
                    if len(args) == 1:
                        matrix.rotate(args[0])
                    else:
                        matrix.translate(args[1], args[2])
                        matrix.rotate(args[0])
                        matrix.translate(-args[1], -args[2])
                elif transform_type == 'scale':
                    if len(args) == 1:
                        matrix.scale(args[0], args[0])
                    else:
                        matrix.scale(args[0], args[1])
                elif transform_type == 'matrix':
                    matrix.matrix(args)

        # 应用变换到路径数据
        if element.nodeName == 'path':
            self.transform_path(element, matrix, 0, 1)
            element.removeAttribute('transform')

        # 递归处理子元素
        for child in element.childNodes:
            self.apply_transform(child)

    def flatten(self, element):
        """展平SVG结构，移除g元素"""
        if not element or element.nodeType != element.ELEMENT_NODE:
            return

        parent = element.parentNode
        if element.nodeName == 'g':
            # 将g元素的属性应用到子元素
            for child in element.childNodes[:]:
                if child.nodeType == child.ELEMENT_NODE:
                    for attr in element.attributes.keys():
                        if not child.hasAttribute(attr):
                            child.setAttribute(attr, element.getAttribute(attr))
                    parent.insertBefore(child, element)
                    self.flatten(child)
            parent.removeChild(element)
        else:
            for child in element.childNodes[:]:
                self.flatten(child)

    def filter(self, allowed_elements):
        """过滤非允许的元素"""
        if not self.svg_root:
            return

        to_remove = []
        for child in self.svg_root.childNodes:
            if (child.nodeType == child.ELEMENT_NODE and
                    child.nodeName not in allowed_elements):
                to_remove.append(child)

        for element in to_remove:
            element.parentNode.removeChild(element)

    def recurse(self, element, callback):
        """递归处理元素"""
        if not element:
            return

        if element.nodeType == element.ELEMENT_NODE:
            callback(element)

        for child in element.childNodes[:]:
            self.recurse(child, callback)

    def get_style(self):
        """获取样式节点"""
        if not self.svg_root:
            return None

        for child in self.svg_root.childNodes:
            if (child.nodeType == child.ELEMENT_NODE and
                    child.tagName == 'style'):
                return child

        return None

    def getStyle(self):  # 添加这个别名方法以保持兼容性
        """获取样式节点（兼容性方法）"""
        return self.get_style()

    def config(self, config: Dict):
        """配置解析器参数"""
        if 'tolerance' in config and not GeometryUtil.almost_equal(float(config['tolerance']), 0):
            self.conf['tolerance'] = float(config['tolerance'])

    def load(self, svg_string: str):
        """加载并解析SVG字符串"""
        print("SvgParser.load: Starting...")
        if not svg_string or not isinstance(svg_string, str):
            print("SvgParser.load: Invalid SVG string")
            raise ValueError('invalid SVG string')

        print(f"SvgParser.load: Parsing XML string of length {len(svg_string)}")
        try:
            self.svg = minidom.parseString(svg_string)
            print("SvgParser.load: XML parsed successfully")
        except Exception as e:
            print(f"SvgParser.load: XML parsing error: {e}")
            raise

        self.svg_root = None

        print("SvgParser.load: Looking for SVG root element")
        for child in self.svg.childNodes:
            if child.nodeType == child.ELEMENT_NODE and child.tagName == 'svg':
                self.svg_root = child
                print("SvgParser.load: Found SVG root element")
                break

        if not self.svg_root:
            print("SvgParser.load: No SVG root element found")
            raise ValueError("SVG has no valid root element")

        print("SvgParser.load: Complete, returning SVG root")
        return self.svg_root

    def clean(self):
        """清理并返回处理后的SVG"""
        print("SvgParser.clean: Starting...")
        if not self.svg_root:
            print("SvgParser.clean: No SVG root to clean")
            return None

        print("SvgParser.clean: Calling clean_input")
        try:
            self.clean_input()
            print("SvgParser.clean: clean_input completed successfully")
        except Exception as e:
            print(f"SvgParser.clean: Error in clean_input: {e}")
            import traceback
            traceback.print_exc()

        print("SvgParser.clean: Complete, returning SVG root")
        return self.svg_root

    def path_to_absolute(self, path_element):
        """将路径转换为绝对坐标"""
        if not path_element or path_element.nodeName != 'path':
            raise ValueError('invalid path')

        d = path_element.getAttribute('d')
        if not d:
            return

        # 使用svg.path解析路径
        path = parse_path(d)

        # 转换为绝对坐标
        absolute_path = []
        current_pos = complex(0, 0)

        for segment in path:
            if isinstance(segment, Line):
                absolute_path.append(f'M {segment.start.real},{segment.start.imag}')
                absolute_path.append(f'L {segment.end.real},{segment.end.imag}')
                current_pos = segment.end
            elif isinstance(segment, CubicBezier):
                absolute_path.append(f'M {segment.start.real},{segment.start.imag}')
                absolute_path.append(
                    f'C {segment.control1.real},{segment.control1.imag} '
                    f'{segment.control2.real},{segment.control2.imag} '
                    f'{segment.end.real},{segment.end.imag}'
                )
                current_pos = segment.end
            elif isinstance(segment, QuadraticBezier):
                absolute_path.append(f'M {segment.start.real},{segment.start.imag}')
                absolute_path.append(
                    f'Q {segment.control.real},{segment.control.imag} '
                    f'{segment.end.real},{segment.end.imag}'
                )
                current_pos = segment.end
            elif isinstance(segment, Arc):
                absolute_path.append(f'M {segment.start.real},{segment.start.imag}')
                absolute_path.append(
                    f'A {segment.radius.real},{segment.radius.imag} '
                    f'{segment.rotation} {int(segment.large_arc)},{int(segment.sweep)} '
                    f'{segment.end.real},{segment.end.imag}'
                )
                current_pos = segment.end

        path_element.setAttribute('d', ' '.join(absolute_path))

    def transform_path(self, element, transform: Matrix, rotate: float, scale: float):
        """变换路径元素"""
        d = element.getAttribute('d')
        if not d:
            return

        # 使用svgpathtools进行更精确的路径处理
        path = parse_path_complex(d)
        transformed_path = []

        for segment in path:
            if segment.is_line():
                start = transform.calc(segment.start.real, segment.start.imag)
                end = transform.calc(segment.end.real, segment.end.imag)
                transformed_path.append(f'M {start[0]},{start[1]} L {end[0]},{end[1]}')

            elif segment.is_cubic():
                start = transform.calc(segment.start.real, segment.start.imag)
                c1 = transform.calc(segment.control1.real, segment.control1.imag)
                c2 = transform.calc(segment.control2.real, segment.control2.imag)
                end = transform.calc(segment.end.real, segment.end.imag)
                transformed_path.append(
                    f'M {start[0]},{start[1]} '
                    f'C {c1[0]},{c1[1]} {c2[0]},{c2[1]} {end[0]},{end[1]}'
                )

            elif segment.is_quadratic():
                start = transform.calc(segment.start.real, segment.start.imag)
                control = transform.calc(segment.control.real, segment.control.imag)
                end = transform.calc(segment.end.real, segment.end.imag)
                transformed_path.append(
                    f'M {start[0]},{start[1]} '
                    f'Q {control[0]},{control[1]} {end[0]},{end[1]}'
                )

            elif segment.is_arc():
                # 对于圆弧，需要调整半径和旋转角度
                start = transform.calc(segment.start.real, segment.start.imag)
                end = transform.calc(segment.end.real, segment.end.imag)
                rx = segment.radius.real * scale
                ry = segment.radius.imag * scale
                phi = segment.rotation + rotate
                transformed_path.append(
                    f'M {start[0]},{start[1]} '
                    f'A {rx},{ry} {phi} '
                    f'{int(segment.large_arc)},{int(segment.sweep)} '
                    f'{end[0]},{end[1]}'
                )

        element.setAttribute('d', ' '.join(transformed_path))

    def split_path(self, path):
        """分割复合路径"""
        if (not path or
                path.nodeType != path.ELEMENT_NODE or
                path.nodeName != 'path' or
                not path.parentNode):
            return False

        d = path.getAttribute('d')
        if not d:
            return False

        # 使用正则表达式找到所有的移动命令
        move_commands = re.finditer(r'[Mm]', d)
        indices = [m.start() for m in move_commands]

        if len(indices) <= 1:
            return False

        # 为每个子路径创建新的路径元素
        parent = path.parentNode
        for i in range(len(indices)):
            start = indices[i]
            end = indices[i + 1] if i + 1 < len(indices) else len(d)

            sub_path = self.svg.createElement('path')
            sub_path.setAttribute('d', d[start:end])

            # 复制原始路径的其他属性
            for attr in path.attributes.keys():
                if attr != 'd':
                    sub_path.setAttribute(attr, path.getAttribute(attr))

            parent.insertBefore(sub_path, path)

        parent.removeChild(path)
        return True

    def linearize_bezier(self, points: List[Dict], tolerance: float) -> List[Dict]:
        """将贝塞尔曲线线性化"""
        if len(points) < 2:
            return points

        def get_point(t: float) -> Dict:
            if len(points) == 4:  # 三次贝塞尔
                mt = 1 - t
                return {
                    'x': mt ** 3 * points[0]['x'] + 3 * mt ** 2 * t * points[1]['x'] +
                         3 * mt * t ** 2 * points[2]['x'] + t ** 3 * points[3]['x'],
                    'y': mt ** 3 * points[0]['y'] + 3 * mt ** 2 * t * points[1]['y'] +
                         3 * mt * t ** 2 * points[2]['y'] + t ** 3 * points[3]['y']
                }
            else:  # 二次贝塞尔
                mt = 1 - t
                return {
                    'x': mt ** 2 * points[0]['x'] + 2 * mt * t * points[1]['x'] +
                         t ** 2 * points[2]['x'],
                    'y': mt ** 2 * points[0]['y'] + 2 * mt * t * points[1]['y'] +
                         t ** 2 * points[2]['y']
                }

        def get_max_distance(start_t: float, end_t: float) -> Tuple[float, Dict]:
            mid_t = (start_t + end_t) / 2
            mid_point = get_point(mid_t)

            start_point = get_point(start_t)
            end_point = get_point(end_t)

            # 计算点到线段的距离
            dx = end_point['x'] - start_point['x']
            dy = end_point['y'] - start_point['y']

            if dx == 0 and dy == 0:
                return 0, mid_point

            t = ((mid_point['x'] - start_point['x']) * dx +
                 (mid_point['y'] - start_point['y']) * dy) / (dx * dx + dy * dy)

            if t < 0:
                t = 0
            elif t > 1:
                t = 1

            proj_x = start_point['x'] + t * dx
            proj_y = start_point['y'] + t * dy

            dist = math.sqrt((mid_point['x'] - proj_x) ** 2 +
                             (mid_point['y'] - proj_y) ** 2)

            return dist, mid_point

        def recursive_linearize(start_t: float, end_t: float,
                                result: List[Dict]) -> None:
            max_dist, mid_point = get_max_distance(start_t, end_t)

            if max_dist <= tolerance:
                result.append(get_point(end_t))
            else:
                mid_t = (start_t + end_t) / 2
                recursive_linearize(start_t, mid_t, result)
                recursive_linearize(mid_t, end_t, result)

        result = [points[0]]
        recursive_linearize(0, 1, result)
        return result

    def linearize_arc(self, start: Dict, end: Dict, rx: float, ry: float,
                      angle: float, large_arc: bool, sweep: bool,
                      tolerance: float) -> List[Dict]:
        """将圆弧线性化"""
        # 将角度转换为弧度
        angle_rad = math.radians(angle)

        # 计算中心点和起始/结束角度
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        # 将端点转换到椭圆坐标系
        dx = (start['x'] - end['x']) / 2
        dy = (start['y'] - end['y']) / 2

        x1 = cos_angle * dx + sin_angle * dy
        y1 = -sin_angle * dx + cos_angle * dy

        # 计算中心点
        ratio = rx / ry
        x1_sq = x1 * x1
        y1_sq = y1 * y1
        rx_sq = rx * rx
        ry_sq = ry * ry

        rad_check = x1_sq / rx_sq + y1_sq / ry_sq
        if rad_check > 1:
            rx = math.sqrt(rad_check) * rx
            ry = math.sqrt(rad_check) * ry
            rx_sq = rx * rx
            ry_sq = ry * ry

        denom = rx_sq * y1_sq + ry_sq * x1_sq
        if denom == 0:
            return [start, end]

        factor = math.sqrt(abs((rx_sq * ry_sq - denom) / denom))
        if large_arc == sweep:
            factor = -factor

        cxp = factor * rx * y1 / ry
        cyp = -factor * ry * x1 / rx

        cx = cos_angle * cxp - sin_angle * cyp + (start['x'] + end['x']) / 2
        cy = sin_angle * cxp + cos_angle * cyp + (start['y'] + end['y']) / 2

        # 计算角度
        start_angle = math.atan2((y1 - cyp) / ry, (x1 - cxp) / rx)
        delta_angle = math.atan2((-y1 - cyp) / ry, (-x1 - cxp) / rx) - start_angle

        if sweep and delta_angle < 0:
            delta_angle += 2 * math.pi
        elif not sweep and delta_angle > 0:
            delta_angle -= 2 * math.pi

        # 线性化
        num_segments = max(
            math.ceil(abs(delta_angle) / math.acos(1 - tolerance / max(rx, ry))),
            4
        )

        result = [start]
        for i in range(1, num_segments):
            t = i / num_segments
            angle = start_angle + t * delta_angle
            cos_t = math.cos(angle)
            sin_t = math.sin(angle)

            x = cos_angle * rx * cos_t - sin_angle * ry * sin_t + cx
            y = sin_angle * rx * cos_t + cos_angle * ry * sin_t + cy

            result.append({'x': x, 'y': y})

        result.append(end)
        return result

    def polygonify(self, element) -> List[Dict]:
        """将SVG元素转换为多边形点列表"""
        if not element:
            return []

        points = []

        if element.nodeName in ['polygon', 'polyline']:
            point_list = element.getAttribute('points').strip().split()
            for i in range(0, len(point_list), 2):
                points.append({
                    'x': float(point_list[i]),
                    'y': float(point_list[i + 1])
                })

        elif element.nodeName == 'rect':
            x = float(element.getAttribute('x') or 0)
            y = float(element.getAttribute('y') or 0)
            width = float(element.getAttribute('width'))
            height = float(element.getAttribute('height'))

            points = [
                {'x': x, 'y': y},
                {'x': x + width, 'y': y},
                {'x': x + width, 'y': y + height},
                {'x': x, 'y': y + height}
            ]

        elif element.nodeName == 'circle':
            cx = float(element.getAttribute('cx'))
            cy = float(element.getAttribute('cy'))
            r = float(element.getAttribute('r'))

            num_segments = max(
                math.ceil((2 * math.pi) / math.acos(1 - (self.conf['tolerance'] / r))),
                3
            )

            for i in range(num_segments):
                theta = i * (2 * math.pi / num_segments)
                points.append({
                    'x': cx + r * math.cos(theta),
                    'y': cy + r * math.sin(theta)
                })

        elif element.nodeName == 'ellipse':
            cx = float(element.getAttribute('cx'))
            cy = float(element.getAttribute('cy'))
            rx = float(element.getAttribute('rx'))
            ry = float(element.getAttribute('ry'))

            max_radius = max(rx, ry)
            num_segments = max(
                math.ceil((2 * math.pi) / math.acos(1 - (self.conf['tolerance'] / max_radius))),
                3
            )

            for i in range(num_segments):
                theta = i * (2 * math.pi / num_segments)
                points.append({
                    'x': cx + rx * math.cos(theta),
                    'y': cy + ry * math.sin(theta)
                })

        elif element.nodeName == 'path':
            d = element.getAttribute('d')
            if not d:
                return points

            path = parse_path(d)
            current_pos = complex(0, 0)

            for segment in path:
                if isinstance(segment, Line):
                    points.extend([
                        {'x': segment.start.real, 'y': segment.start.imag},
                        {'x': segment.end.real, 'y': segment.end.imag}
                    ])
                    current_pos = segment.end

                elif isinstance(segment, (CubicBezier, QuadraticBezier)):
                    control_points = []
                    if isinstance(segment, CubicBezier):
                        control_points = [
                            {'x': segment.start.real, 'y': segment.start.imag},
                            {'x': segment.control1.real, 'y': segment.control1.imag},
                            {'x': segment.control2.real, 'y': segment.control2.imag},
                            {'x': segment.end.real, 'y': segment.end.imag}
                        ]
                    else:
                        control_points = [
                            {'x': segment.start.real, 'y': segment.start.imag},
                            {'x': segment.control.real, 'y': segment.control.imag},
                            {'x': segment.end.real, 'y': segment.end.imag}
                        ]
                    points.extend(self.linearize_bezier(control_points, self.conf['tolerance']))
                    current_pos = segment.end

                elif isinstance(segment, Arc):
                    points.extend(self.linearize_arc(
                        {'x': segment.start.real, 'y': segment.start.imag},
                        {'x': segment.end.real, 'y': segment.end.imag},
                        segment.radius.real,
                        segment.radius.imag,
                        segment.rotation,
                        segment.large_arc,
                        segment.sweep,
                        self.conf['tolerance']
                    ))
                    current_pos = segment.end

        # 移除重复的端点
        while (len(points) > 0 and
               GeometryUtil.almost_equal(points[0]['x'], points[-1]['x'], self.conf['toleranceSvg']) and
               GeometryUtil.almost_equal(points[0]['y'], points[-1]['y'], self.conf['toleranceSvg'])):
            points.pop()

        return points

    def launch_workers(self, tree, bin_polygon, config, progress_callback, display_callback):
        """启动工作进程"""
        print("launch_workers: Starting...")

        # 添加更多的调试信息
        print(f"Tree size: {len(tree) if tree else 0}")
        print(f"Bin polygon has {len(bin_polygon) if bin_polygon else 0} points")

        try:
            # 这里添加工作进程的逻辑
            print("launch_workers: Initializing worker...")

            # 添加代码，确保不会发生未捕获的异常

            print("launch_workers: Complete")
        except Exception as e:
            print(f"Error in launch_workers: {e}")
            import traceback
            traceback.print_exc()
            self.working = False
