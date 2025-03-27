#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SvgNest Python Implementation
Python port of SvgNest algorithm for optimizing 2D shape nesting
"""

import math
import copy
import random
import json
from typing import List, Dict, Tuple, Any, Optional, Union  # 明确导入typing模块
import threading
import time
import pyclipper  # 使用pyclipper代替clipper库
# 确保正确导入其他需要的模块
import geometryutil
from geometryutil import GeometryUtil
import svgparser
import placementworker
import matrix

class SvgNest:
    """Main class for the SvgNest algorithm"""

    def __init__(self):
        self.svg = None
        self.style = None
        self.parts = None
        self.tree = None
        self.bin = None
        self.bin_polygon = None
        self.bin_bounds = None
        self.nfp_cache = {}
        self.config = {
            'clipper_scale': 10000000,
            'curve_tolerance': 0.3,
            'spacing': 0,
            'rotations': 4,
            'population_size': 10,
            'mutation_rate': 10,
            'use_holes': False,
            'explore_concave': False
        }
        self.working = False
        self.GA = None
        self.best = None
        self.worker_timer = None
        self.progress = 0

    def parse_svg(self, svg_string: str) -> Any:
        """Parse an SVG string and prepare it for nesting"""
        # Reset if in progress
        self.stop()

        self.bin = None
        self.bin_polygon = None
        self.tree = None

        # Parse SVG using the converted SvgParser library
        self.svg = svgparser.load(svg_string)
        self.style = svgparser.getStyle()
        self.svg = svgparser.clean()

        if self.svg is not None:  # 明确检查是否有效
            self.tree = self.get_parts(self.svg.childNodes)

        return self.svg

    def set_bin(self, element: Any) -> None:
        """Set the bin element for nesting"""
        if not self.svg:
            return
        self.bin = element

    def config(self, c: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Set or get configuration parameters"""
        if not c:
            return self.config

        # Clean up inputs
        if 'curve_tolerance' in c and not GeometryUtil.almostEqual(float(c['curve_tolerance']), 0):
            self.config['curve_tolerance'] = float(c['curve_tolerance'])

        if 'spacing' in c:
            self.config['spacing'] = float(c['spacing'])

        if 'rotations' in c and int(c['rotations']) > 0:
            self.config['rotations'] = int(c['rotations'])

        if 'population_size' in c and int(c['population_size']) > 2:
            self.config['population_size'] = int(c['population_size'])

        if 'mutation_rate' in c and int(c['mutation_rate']) > 0:
            self.config['mutation_rate'] = int(c['mutation_rate'])

        if 'use_holes' in c:
            self.config['use_holes'] = bool(c['use_holes'])

        if 'explore_concave' in c:
            self.config['explore_concave'] = bool(c['explore_concave'])

        svgparser.config({'tolerance': self.config['curve_tolerance']})

        self.best = None
        self.nfp_cache = {}
        self.bin_polygon = None
        self.GA = None

        return self.config

    def start(self, progress_callback, display_callback) -> bool:
        """Start the nesting process"""
        if not self.svg or not self.bin:
            return False

        self.parts = list(self.svg.childNodes)
        bin_index = self.parts.index(self.bin) if self.bin in self.parts else -1

        if bin_index >= 0:
            # Don't process bin as a part of the tree
            self.parts.pop(bin_index)

        # Build tree without bin
        self.tree = self.get_parts(self.parts.copy())

        # Offset tree with spacing
        self._offset_tree(self.tree, 0.5 * self.config['spacing'], self.polygon_offset)

        self.bin_polygon = svgparser.polygonify(self.bin)
        self.bin_polygon = self.clean_polygon(self.bin_polygon)

        if not self.bin_polygon or len(self.bin_polygon) < 3:
            return False

        self.bin_bounds = GeometryUtil.getPolygonBounds(self.bin_polygon)

        if self.config['spacing'] > 0:
            offset_bin = self.polygon_offset(self.bin_polygon, -0.5 * self.config['spacing'])
            if len(offset_bin) == 1:
                self.bin_polygon = offset_bin[0]

        self.bin_polygon.id = -1

        # Put bin on origin
        x_bin_min = min(point['x'] for point in self.bin_polygon)
        y_bin_min = min(point['y'] for point in self.bin_polygon)

        for i in range(len(self.bin_polygon)):
            self.bin_polygon[i]['x'] -= x_bin_min
            self.bin_polygon[i]['y'] -= y_bin_min

        x_bin_max = max(point['x'] for point in self.bin_polygon)
        y_bin_max = max(point['y'] for point in self.bin_polygon)

        self.bin_polygon.width = x_bin_max - x_bin_min
        self.bin_polygon.height = y_bin_max - y_bin_min

        # All paths need to have the same winding direction
        if GeometryUtil.polygonArea(self.bin_polygon) > 0:
            self.bin_polygon.reverse()

        # Remove duplicate endpoints, ensure counterclockwise winding direction
        for i in range(len(self.tree)):
            start = self.tree[i][0]
            end = self.tree[i][-1]

            if start == end or (GeometryUtil.almostEqual(start['x'], end['x']) and
                                GeometryUtil.almostEqual(start['y'], end['y'])):
                self.tree[i].pop()

            if GeometryUtil.polygonArea(self.tree[i]) > 0:
                self.tree[i].reverse()

        self.working = False

        # Use a thread to simulate the interval in JavaScript
        def worker_function():
            while self.working:
                progress_callback(self.progress)
                time.sleep(0.1)  # 100ms interval as in original JS

        self.working = True
        self.worker_timer = threading.Thread(target=worker_function)
        self.worker_timer.daemon = True
        self.worker_timer.start()

        # Launch workers to start the nesting process
        self.launch_workers(self.tree, self.bin_polygon, self.config, progress_callback, display_callback)

        return True

    def _offset_tree(self, t: List[Any], offset: float, offset_function: callable) -> None:
        """Offset a tree of polygons recursively"""
        for i in range(len(t)):
            offset_paths = offset_function(t[i], offset)
            if len(offset_paths) == 1:
                # Replace array items in place
                t[i][:] = offset_paths[0]

            if hasattr(t[i], 'childNodes') and len(t[i].childNodes) > 0:
                self._offset_tree(t[i].childNodes, -offset, offset_function)

    def launch_workers(self, tree: List[Any], bin_polygon: List[Dict[str, float]],
                       config: Dict[str, Any], progress_callback, display_callback) -> None:
        """Launch workers for placement calculation"""

        def shuffle(array: List[Any]) -> List[Any]:
            """Shuffle an array in place"""
            array_copy = array.copy()
            for i in range(len(array_copy) - 1, 0, -1):
                j = random.randint(0, i)
                array_copy[i], array_copy[j] = array_copy[j], array_copy[i]
            return array_copy

        if self.GA is None:
            # Initiate new GA
            adam = tree.copy()

            # Seed with decreasing area
            adam.sort(key=lambda x: -abs(GeometryUtil.polygonArea(x)))

            self.GA = GeneticAlgorithm(adam, bin_polygon, config)

        individual = None

        # Evaluate all members of the population
        for i in range(len(self.GA.population)):
            if not hasattr(self.GA.population[i], 'fitness') or self.GA.population[i].fitness is None:
                individual = self.GA.population[i]
                break

        if individual is None:
            # All individuals have been evaluated, start next generation
            self.GA.generation()
            individual = self.GA.population[1]

        place_list = individual.placement
        rotations = individual.rotation

        ids = [place_list[i].id for i in range(len(place_list))]

        nfp_pairs = []
        new_cache = {}

        for i in range(len(place_list)):
            part = place_list[i]
            key = {'A': bin_polygon.id, 'B': part.id, 'inside': True,
                   'Arotation': 0, 'Brotation': rotations[i]}
            key_str = json.dumps(key)

            if key_str not in self.nfp_cache:
                nfp_pairs.append({'A': bin_polygon, 'B': part, 'key': key})
            else:
                new_cache[key_str] = self.nfp_cache[key_str]

            for j in range(i):
                placed = place_list[j]
                key = {'A': placed.id, 'B': part.id, 'inside': False,
                       'Arotation': rotations[j], 'Brotation': rotations[i]}
                key_str = json.dumps(key)

                if key_str not in self.nfp_cache:
                    nfp_pairs.append({'A': placed, 'B': part, 'key': key})
                else:
                    new_cache[key_str] = self.nfp_cache[key_str]

        # Only keep cache for one cycle
        self.nfp_cache = new_cache

        worker = placementworker.PlacementWorker(bin_polygon, place_list.copy(), ids, rotations, config, self.nfp_cache)

        # Process NFP pairs - in Python we can use multiprocessing for this
        # For simplicity, we'll process them sequentially for now
        self.progress = 0
        spawn_count = 0
        generated_nfp = []

        # Function to process a single NFP pair
        def process_nfp_pair(pair):
            nonlocal spawn_count
            spawn_count += 1
            self.progress = spawn_count / len(nfp_pairs)

            if not pair:
                return None

            search_edges = config['explore_concave']
            use_holes = config['use_holes']

            A = GeometryUtil.rotatePolygon(pair['A'], pair['key']['Arotation'])
            B = GeometryUtil.rotatePolygon(pair['B'], pair['key']['Brotation'])

            nfp = None

            if pair['key']['inside']:
                if GeometryUtil.isRectangle(A, 0.001):
                    nfp = GeometryUtil.noFitPolygonRectangle(A, B)
                else:
                    nfp = GeometryUtil.noFitPolygon(A, B, True, search_edges)

                # Ensure all interior NFPs have the same winding direction
                if nfp and len(nfp) > 0:
                    for i in range(len(nfp)):
                        if GeometryUtil.polygonArea(nfp[i]) > 0:
                            nfp[i].reverse()
                else:
                    # Warning on null inner NFP
                    print(f"NFP Warning: {pair['key']}")
            else:
                if search_edges:
                    nfp = GeometryUtil.noFitPolygon(A, B, False, search_edges)
                else:
                    nfp = minkowski_difference(A, B)

                # Sanity check
                if not nfp or len(nfp) == 0:
                    print(f"NFP Error: {pair['key']}")
                    print(f"A: {json.dumps(A)}")
                    print(f"B: {json.dumps(B)}")
                    return None

                for i in range(len(nfp)):
                    if not search_edges or i == 0:  # Only the first NFP is guaranteed to pass sanity check
                        if abs(GeometryUtil.polygonArea(nfp[i])) < abs(GeometryUtil.polygonArea(A)):
                            print(f"NFP Area Error: {abs(GeometryUtil.polygonArea(nfp[i]))}, {pair['key']}")
                            print(f"NFP: {json.dumps(nfp[i])}")
                            print(f"A: {json.dumps(A)}")
                            print(f"B: {json.dumps(B)}")
                            nfp.pop(i)
                            return None

                if len(nfp) == 0:
                    return None

                # For outer NFPs, the first is guaranteed to be the largest
                for i in range(len(nfp)):
                    if GeometryUtil.polygonArea(nfp[i]) > 0:
                        nfp[i].reverse()

                    if i > 0:
                        if GeometryUtil.pointInPolygon(nfp[i][0], nfp[0]):
                            if GeometryUtil.polygonArea(nfp[i]) < 0:
                                nfp[i].reverse()

                # Generate NFPs for children (holes of parts) if any exist
                if use_holes and hasattr(A, 'childNodes') and len(A.childNodes) > 0:
                    B_bounds = GeometryUtil.getPolygonBounds(B)

                    for i in range(len(A.childNodes)):
                        A_bounds = GeometryUtil.getPolygonBounds(A.childNodes[i])

                        # No need to find NFP if B's bounding box is too big
                        if A_bounds['width'] > B_bounds['width'] and A_bounds['height'] > B_bounds['height']:
                            cnfp = GeometryUtil.noFitPolygon(A.childNodes[i], B, True, search_edges)

                            # Ensure all interior NFPs have the same winding direction
                            if cnfp and len(cnfp) > 0:
                                for j in range(len(cnfp)):
                                    if GeometryUtil.polygonArea(cnfp[j]) < 0:
                                        cnfp[j].reverse()
                                    nfp.append(cnfp[j])

            return {'key': pair['key'], 'value': nfp}

        # Process NFP pairs
        for pair in nfp_pairs:
            result = process_nfp_pair(pair)
            if result:
                generated_nfp.append(result)

        # Update NFP cache with generated NFPs
        if generated_nfp:
            for nfp in generated_nfp:
                if nfp:
                    key_str = json.dumps(nfp['key'])
                    self.nfp_cache[key_str] = nfp['value']

        worker.nfp_cache = self.nfp_cache

        # Place paths
        placements = [worker.placePaths([place_list.copy()])]

        if not placements or len(placements) == 0:
            return

        individual.fitness = placements[0]['fitness']
        best_result = placements[0]

        for i in range(1, len(placements)):
            if placements[i]['fitness'] < best_result['fitness']:
                best_result = placements[i]

        if not self.best or best_result['fitness'] < self.best['fitness']:
            self.best = best_result

            placed_area = 0
            total_area = 0
            num_parts = len(place_list)
            num_placed_parts = 0

            for i in range(len(self.best['placements'])):
                total_area += abs(GeometryUtil.polygonArea(bin_polygon))
                for j in range(len(self.best['placements'][i])):
                    placed_area += abs(GeometryUtil.polygonArea(
                        tree[self.best['placements'][i][j].id]
                    ))
                    num_placed_parts += 1

            display_callback(
                self.apply_placement(self.best['placements']),
                placed_area / total_area,
                num_placed_parts,
                num_parts
            )
        else:
            display_callback()

        self.working = False

    def get_parts(self, paths: List[Any]) -> List[List[Dict[str, float]]]:
        """Get parts from paths as polygons"""
        polygons = []

        for i in range(len(paths)):
            poly = svgparser.polygonify(paths[i])
            poly = self.clean_polygon(poly)

            if (poly and len(poly) > 2 and
                    abs(GeometryUtil.polygonArea(poly)) >
                    self.config['curve_tolerance'] * self.config['curve_tolerance']):
                poly.source = i
                polygons.append(poly)

        # Turn the list into a tree
        self._to_tree(polygons)

        return polygons

    def _to_tree(self, polygon_list: List[Any], id_start: int = 0) -> int:
        """Convert a list of polygons to a tree structure"""
        parents = []

        # Assign a unique id to each leaf
        id_counter = id_start

        for i in range(len(polygon_list)):
            p = polygon_list[i]

            is_child = False
            for j in range(len(polygon_list)):
                if j == i:
                    continue

                if GeometryUtil.pointInPolygon(p[0], polygon_list[j]):
                    if not hasattr(polygon_list[j], 'children'):
                        polygon_list[j].children = []

                    polygon_list[j].children.append(p)
                    p.parent = polygon_list[j]
                    is_child = True
                    break

            if not is_child:
                parents.append(p)

        # Remove children from the list
        i = 0
        while i < len(polygon_list):
            if polygon_list[i] not in parents:
                polygon_list.pop(i)
            else:
                i += 1

        for i in range(len(parents)):
            parents[i].id = id_counter
            id_counter += 1

        for i in range(len(parents)):
            if hasattr(parents[i], 'children') and parents[i].children:
                id_counter = self._to_tree(parents[i].children, id_counter)

        return id_counter

    def polygon_offset(self, polygon: List[Dict[str, float]], offset: float) -> List[List[Dict[str, float]]]:
        """Offset a polygon by a given amount"""
        if not offset or offset == 0 or GeometryUtil.almostEqual(offset, 0):
            return [polygon]

        # 使用PyClipper库执行多边形偏移
        p = self.svg_to_clipper(polygon)

        # 创建PyClipper偏移对象
        miter_limit = 2
        co = pyclipper.PyclipperOffset(miter_limit, self.config['curve_tolerance'] * self.config['clipper_scale'])
        co.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        # 执行偏移并获取结果
        new_paths = co.Execute(offset * self.config['clipper_scale'])

        result = []
        for i in range(len(new_paths)):
            result.append(self.clipper_to_svg(new_paths[i]))

        return result

    def clean_polygon(self, polygon: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Clean a polygon by removing self-intersections and smoothing"""
        if not polygon or len(polygon) < 3:
            return None

        p = self.svg_to_clipper(polygon)

        # 使用PyClipper简化多边形，去除自相交
        pc = pyclipper.Pyclipper()
        pc.AddPath(p, pyclipper.PT_SUBJECT, True)
        simple = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

        if not simple or len(simple) == 0:
            return None

        # 找到面积最大的多边形
        biggest = simple[0]
        biggest_area = abs(pyclipper.Area(biggest))

        for i in range(1, len(simple)):
            area = abs(pyclipper.Area(simple[i]))
            if area > biggest_area:
                biggest = simple[i]
                biggest_area = area

        # 清理奇异点、重合点和边
        clean = pyclipper.CleanPolygon(
            biggest, self.config['curve_tolerance'] * self.config['clipper_scale']
        )

        if not clean or len(clean) == 0:
            return None

        return self.clipper_to_svg(clean)

    def svg_to_clipper(self, polygon: List[Dict[str, float]]) -> List[List[int]]:
        """Convert SVG coordinates to Clipper coordinates"""
        # PyClipper需要的是整数坐标，所以先放大再取整
        clip = []
        for point in polygon:
            clip.append([
                int(point['x'] * self.config['clipper_scale']),
                int(point['y'] * self.config['clipper_scale'])
            ])

        return clip

    def clipper_to_svg(self, polygon: List[List[int]]) -> List[Dict[str, float]]:
        """Convert Clipper coordinates to SVG coordinates"""
        normal = []
        for point in polygon:
            normal.append({
                'x': point[0] / self.config['clipper_scale'],
                'y': point[1] / self.config['clipper_scale']
            })

        return normal

    def apply_placement(self, placement: List[List[Dict[str, Any]]]) -> List[Any]:
        """Apply the placement to SVG elements"""
        from lxml import etree
        from copy import deepcopy

        # 创建深拷贝而不是使用cloneNode
        clone = []
        for i in range(len(self.parts)):
            clone.append(deepcopy(self.parts[i]))

        svg_list = []

        for i in range(len(placement)):
            # 创建新的SVG元素
            new_svg = deepcopy(self.svg)

            # 设置viewBox和尺寸
            new_svg.set('viewBox', f'0 0 {self.bin_bounds["width"]} {self.bin_bounds["height"]}')
            new_svg.set('width', f'{self.bin_bounds["width"]}px')
            new_svg.set('height', f'{self.bin_bounds["height"]}px')

            # 克隆bin元素
            bin_clone = deepcopy(self.bin)
            bin_clone.set('class', 'bin')
            bin_clone.set('transform', f'translate({-self.bin_bounds["x"]} {-self.bin_bounds["y"]})')
            new_svg.append(bin_clone)

            for j in range(len(placement[i])):
                p = placement[i][j]
                part = self.tree[p.id]

                # 创建变换组
                part_group = etree.Element('{http://www.w3.org/2000/svg}g')
                part_group.set('transform', f'translate({p.x} {p.y}) rotate({p.rotation})')
                part_group.append(clone[part.source])

                if hasattr(part, 'children') and len(part.children) > 0:
                    flattened = self._flatten_tree(part.children, True)
                    for k in range(len(flattened)):
                        c = clone[flattened[k].source]
                        # 添加类以指示孔洞
                        if flattened[k].hole:
                            class_attr = c.get('class', '')
                            if 'hole' not in class_attr:
                                c.set('class', (class_attr + ' hole').strip())
                        part_group.append(c)

                new_svg.append(part_group)

            svg_list.append(new_svg)

        return svg_list

    def _flatten_tree(self, t: List[Any], hole: bool) -> List[Any]:
        """Flatten a tree into a list"""
        flat = []
        for i in range(len(t)):
            flat.append(t[i])
            t[i].hole = hole
            if hasattr(t[i], 'children') and len(t[i].children) > 0:
                flat.extend(self._flatten_tree(t[i].children, not hole))

        return flat

    def stop(self) -> None:
        """Stop the nesting process"""
        self.working = False
        if self.worker_timer and self.worker_timer.is_alive():
            # Can't directly stop a thread in Python, but setting working to False
            # will make it exit on next iteration
            self.worker_timer = None


def minkowski_difference(A: List[Dict[str, float]], B: List[Dict[str, float]]) -> List[List[Dict[str, float]]]:
    """Calculate the Minkowski difference between polygons A and B"""
    # 将A和B转换为PyClipper可用的坐标格式（整数坐标）
    scale = 10000000  # 坐标缩放因子

    # 转换A到PyClipper格式
    Ac = []
    for point in A:
        Ac.append([int(point['x'] * scale), int(point['y'] * scale)])

    # 转换B到PyClipper格式并取负值（Minkowski差需要B取负）
    Bc = []
    for point in B:
        Bc.append([int(-point['x'] * scale), int(-point['y'] * scale)])

    # 使用PyClipper计算Minkowski和
    solution = pyclipper.MinkowskiSum(Ac, Bc, True)
    clipper_nfp = None

    # 找出面积最大的多边形
    largest_area = None
    for i in range(len(solution)):
        n = []
        for point in solution[i]:
            n.append({'x': point[0] / scale, 'y': point[1] / scale})

        sarea = GeometryUtil.polygonArea(n)
        if largest_area is None or largest_area > sarea:
            clipper_nfp = n
            largest_area = sarea

    # 将B[0]添加到所有点
    for i in range(len(clipper_nfp)):
        clipper_nfp[i]['x'] += B[0]['x']
        clipper_nfp[i]['y'] += B[0]['y']

    return [clipper_nfp]


class GeneticAlgorithm:
    """Genetic algorithm for optimizing nesting"""

    def __init__(self, adam: List[Any], bin_polygon: List[Dict[str, float]], config: Dict[str, Any]):
        self.config = config or {
            'population_size': 10,
            'mutation_rate': 10,
            'rotations': 4
        }
        self.bin_bounds = GeometryUtil.getPolygonBounds(bin_polygon)

        # Population is an array of individuals
        # Each individual represents the order of insertion and rotation angle
        angles = []
        for i in range(len(adam)):
            angles.append(self.random_angle(adam[i]))

        self.population = [{'placement': adam, 'rotation': angles}]

        while len(self.population) < config['population_size']:
            mutant = self.mutate(self.population[0])
            self.population.append(mutant)

    def random_angle(self, part: List[Dict[str, float]]) -> float:
        """Get a random angle for insertion"""
        angle_list = [i * (360 / max(self.config['rotations'], 1))
                      for i in range(max(self.config['rotations'], 1))]

        # Shuffle the angle list
        random.shuffle(angle_list)

        for angle in angle_list:
            rotated_part = GeometryUtil.rotatePolygon(part, angle)

            # Don't use angles where the part doesn't fit in the bin
            if rotated_part.width < self.bin_bounds['width'] and rotated_part.height < self.bin_bounds['height']:
                return angle

        return 0

    def mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an individual with the given mutation rate"""
        clone = {
            'placement': individual['placement'].copy(),
            'rotation': individual['rotation'].copy()
        }

        for i in range(len(clone['placement'])):
            # Swap with next part with probability based on mutation rate
            if random.random() < 0.01 * self.config['mutation_rate']:
                j = i + 1
                if j < len(clone['placement']):
                    clone['placement'][i], clone['placement'][j] = clone['placement'][j], clone['placement'][i]

            # Change rotation with probability based on mutation rate
            if random.random() < 0.01 * self.config['mutation_rate']:
                clone['rotation'][i] = self.random_angle(clone['placement'][i])

        return clone

    def mate(self, male: Dict[str, Any], female: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Single point crossover between two individuals"""
        cutpoint = round(min(max(random.random(), 0.1), 0.9) * (len(male['placement']) - 1))

        gene1 = male['placement'][:cutpoint]
        rot1 = male['rotation'][:cutpoint]

        gene2 = female['placement'][:cutpoint]
        rot2 = female['rotation'][:cutpoint]

        def contains(gene, id_val):
            """Check if a gene contains an id"""
            for g in gene:
                if g.id == id_val:
                    return True
            return False

        # Complete gene1 with missing elements from female
        for i in range(len(female['placement'])):
            if not contains(gene1, female['placement'][i].id):
                gene1.append(female['placement'][i])
                rot1.append(female['rotation'][i])

        # Complete gene2 with missing elements from male
        for i in range(len(male['placement'])):
            if not contains(gene2, male['placement'][i].id):
                gene2.append(male['placement'][i])
                rot2.append(male['rotation'][i])

        return [
            {'placement': gene1, 'rotation': rot1},
            {'placement': gene2, 'rotation': rot2}
        ]

    def generation(self) -> None:
        """Create a new generation using selection, crossover and mutation"""
        # Sort individuals by fitness (lower is better)
        self.population.sort(key=lambda x: x.fitness if hasattr(x, 'fitness') else float('inf'))

        # Fittest individual is preserved in the new generation (elitism)
        new_population = [self.population[0]]

        while len(new_population) < len(self.population):
            male = self.random_weighted_individual()
            female = self.random_weighted_individual(male)

            # Each mating produces two children
            children = self.mate(male, female)

            # Slightly mutate children
            new_population.append(self.mutate(children[0]))

            if len(new_population) < len(self.population):
                new_population.append(self.mutate(children[1]))

        self.population = new_population

    def random_weighted_individual(self, exclude: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Returns a random individual from the population, weighted to the front of the list"""
        pop = self.population.copy()

        if exclude and exclude in pop:
            pop.remove(exclude)

        rand = random.random()

        lower = 0
        weight = 1 / len(pop)
        upper = weight

        for i in range(len(pop)):
            # If the random number falls between lower and upper bounds, select this individual
            if lower < rand < upper:
                return pop[i]
            lower = upper
            upper += 2 * weight * ((len(pop) - i) / len(pop))

        return pop[0]


def main():
    """Example usage of SvgNest"""
    nest = SvgNest()

    # Load SVG file - replace with actual file loading in your implementation
    with open('input.svg', 'r') as file:
        svg_input = file.read()

    # Parse SVG
    svg = nest.parse_svg(svg_input)

    # Set bin element - in practice you would select the appropriate element
    bin_element = svg['childNodes'][0]  # Example, would be selected in practice
    nest.set_bin(bin_element)

    # Configure nesting parameters
    nest.config.update({
        'spacing': 2,
        'rotations': 4,
        'population_size': 10,
        'mutation_rate': 10,
        'use_holes': True,
        'explore_concave': True
    })

    # Define callbacks for progress and display
    def progress_callback(progress):
        """Called when progress is made"""
        print(f"Progress: {progress * 100:.1f}%")

    def display_callback(placement=None, efficiency=None, placed=None, total=None):
        """Called when a new placement has been made"""
        if placement:
            print(f"New placement found - Efficiency: {efficiency * 100:.1f}%")
            print(f"Placed {placed} out of {total} parts")

            # In a real implementation, you might visualize or save the placement
            # For example, save each SVG to a file
            for i, svg_element in enumerate(placement):
                with open(f'placement_{i}.svg', 'w') as file:
                    # Convert SVG element to string and save
                    file.write(str(svg_element))

    # Start nesting process
    nest.start(progress_callback, display_callback)

    # Wait for completion - in a real application you might have a UI or other mechanism
    # while nest.working:
    #     time.sleep(0.5)

    print("Nesting completed!")

if __name__ == '__main__':
    main()