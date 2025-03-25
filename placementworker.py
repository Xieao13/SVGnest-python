#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Placement Worker Python Library
Python conversion of the JavaScript placement worker
"""

import math
import json
from geometryutil import GeometryUtil
import pyclipper  # Python version of ClipperLib


# jsClipper uses X/Y instead of x/y...
def toClipperCoordinates(polygon):
    clone = []
    for i in range(len(polygon)):
        clone.append({
            'X': polygon[i]['x'],
            'Y': polygon[i]['y']
        })

    return clone


def toNestCoordinates(polygon, scale):
    clone = []
    for i in range(len(polygon)):
        clone.append({
            'x': polygon[i]['X'] / scale,
            'y': polygon[i]['Y'] / scale
        })

    return clone


def rotatePolygon(polygon, degrees):
    rotated = []
    angle = degrees * math.pi / 180
    for i in range(len(polygon)):
        x = polygon[i]['x']
        y = polygon[i]['y']
        x1 = x * math.cos(angle) - y * math.sin(angle)
        y1 = x * math.sin(angle) + y * math.cos(angle)

        rotated.append({'x': x1, 'y': y1})

    if hasattr(polygon, 'children') and len(polygon.children) > 0:
        rotated.children = []
        for j in range(len(polygon.children)):
            rotated.children.append(rotatePolygon(polygon.children[j], degrees))

    return rotated


# 辅助函数：清理多边形
def clean_polygon_points(polygon, distance):
    # 使用 pyclipper 的 SimplifyPolygon 函数代替 clean_polygon
    # 将字典形式的点转换为元组列表
    points = [(pt['X'], pt['Y']) for pt in polygon]
    # 使用 SIMPLIFY_AREA 简化算法
    cleaned = pyclipper.SimplifyPolygon([points], pyclipper.PFT_NONZERO)

    if not cleaned:
        return None

    # 转换回字典格式
    return [{'X': pt[0], 'Y': pt[1]} for pt in cleaned[0]]


class PlacementWorker:
    def __init__(self, binPolygon, paths, ids, rotations, config, nfpCache=None):
        self.binPolygon = binPolygon
        self.paths = paths
        self.ids = ids
        self.rotations = rotations
        self.config = config
        self.nfpCache = nfpCache or {}

    # return a placement for the paths/rotations given
    def placePaths(self, paths):
        global minwidth
        if not self.binPolygon:
            return None

        # rotate paths by given rotation
        rotated = []
        for i in range(len(paths)):
            r = rotatePolygon(paths[i], paths[i]['rotation'])
            r['rotation'] = paths[i]['rotation']
            r['source'] = paths[i]['source']
            r['id'] = paths[i]['id']
            rotated.append(r)

        paths = rotated

        allplacements = []
        fitness = 0
        binarea = abs(GeometryUtil.polygonArea(self.binPolygon))

        while len(paths) > 0:
            placed = []
            placements = []
            fitness += 1  # add 1 for each new bin opened (lower fitness is better)

            for i in range(len(paths)):
                path = paths[i]

                # inner NFP
                key = json.dumps({
                    'A': -1,
                    'B': path['id'],
                    'inside': True,
                    'Arotation': 0,
                    'Brotation': path['rotation']
                })

                binNfp = self.nfpCache.get(key)

                # part unplaceable, skip
                if not binNfp or len(binNfp) == 0:
                    continue

                # ensure all necessary NFPs exist
                error = False
                for j in range(len(placed)):
                    key = json.dumps({
                        'A': placed[j]['id'],
                        'B': path['id'],
                        'inside': False,
                        'Arotation': placed[j]['rotation'],
                        'Brotation': path['rotation']
                    })

                    nfp = self.nfpCache.get(key)

                    if not nfp:
                        error = True
                        break

                # part unplaceable, skip
                if error:
                    continue

                position = None
                if len(placed) == 0:
                    # first placement, put it on the left
                    for j in range(len(binNfp)):
                        for k in range(len(binNfp[j])):
                            if position is None or binNfp[j][k]['x'] - path[0]['x'] < position['x']:
                                position = {
                                    'x': binNfp[j][k]['x'] - path[0]['x'],
                                    'y': binNfp[j][k]['y'] - path[0]['y'],
                                    'id': path['id'],
                                    'rotation': path['rotation']
                                }

                    placements.append(position)
                    placed.append(path)

                    continue

                clipperBinNfp = []
                for j in range(len(binNfp)):
                    clipperBinNfp.append(toClipperCoordinates(binNfp[j]))

                # Scale up for ClipperLib equivalent operations
                pco = pyclipper.Pyclipper()
                pc_scale = self.config['clipperScale']

                # Scale up the clipperBinNfp
                scaled_clipperBinNfp = []
                for poly in clipperBinNfp:
                    scaled_poly = [{'X': int(pt['X'] * pc_scale), 'Y': int(pt['Y'] * pc_scale)} for pt in poly]
                    scaled_clipperBinNfp.append(scaled_poly)

                combinedNfp = []

                for j in range(len(placed)):
                    key = json.dumps({
                        'A': placed[j]['id'],
                        'B': path['id'],
                        'inside': False,
                        'Arotation': placed[j]['rotation'],
                        'Brotation': path['rotation']
                    })

                    nfp = self.nfpCache.get(key)

                    if not nfp:
                        continue

                    for k in range(len(nfp)):
                        clone = toClipperCoordinates(nfp[k])
                        for m in range(len(clone)):
                            clone[m]['X'] += placements[j]['x']
                            clone[m]['Y'] += placements[j]['y']

                        # Scale up for Clipper operations
                        scaled_clone = [{'X': int(pt['X'] * pc_scale), 'Y': int(pt['Y'] * pc_scale)} for pt in clone]

                        # 使用自定义的清理多边形函数，或使用SimplifyPolygon
                        # 转换为元组列表用于pyclipper操作
                        scaled_clone_tuples = [(pt['X'], pt['Y']) for pt in scaled_clone]
                        cleaned_result = pyclipper.SimplifyPolygon([scaled_clone_tuples], pyclipper.PFT_NONZERO)

                        if not cleaned_result:
                            continue

                        # 转换回字典格式
                        scaled_clone = [{'X': pt[0], 'Y': pt[1]} for pt in cleaned_result[0]]

                        # Calculate area (equivalent to ClipperLib.Clipper.Area)
                        # 转换为元组列表计算面积
                        scaled_clone_tuples = [(pt['X'], pt['Y']) for pt in scaled_clone]
                        area = abs(pyclipper.Area(scaled_clone_tuples))

                        if len(scaled_clone) > 2 and area > 0.1 * pc_scale * pc_scale:
                            # Add to clipper operation
                            pco.AddPath(scaled_clone_tuples, pyclipper.PT_SUBJECT, True)

                # Execute union operation
                try:
                    solution = pco.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
                    if not solution:
                        continue
                    combinedNfp = solution
                except:
                    continue

                # Difference with bin polygon
                pco = pyclipper.Pyclipper()

                # Convert combinedNfp to the format expected by pyclipper
                for path in combinedNfp:
                    pco.AddPath(path, pyclipper.PT_CLIP, True)

                # Convert scaled_clipperBinNfp to the format expected by pyclipper
                for poly in scaled_clipperBinNfp:
                    poly_tuples = [(pt['X'], pt['Y']) for pt in poly]
                    pco.AddPath(poly_tuples, pyclipper.PT_SUBJECT, True)

                try:
                    finalNfp = pco.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
                    if not finalNfp:
                        continue
                except:
                    continue

                # Clean polygons
                cleaned_finalNfp = []
                for poly in finalNfp:
                    # 使用SimplifyPolygon代替clean_polygon
                    cleaned = pyclipper.SimplifyPolygon([poly], pyclipper.PFT_NONZERO)
                    if cleaned:
                        cleaned_finalNfp.append(cleaned[0])

                finalNfp = cleaned_finalNfp

                # Filter small polygons
                j = 0
                while j < len(finalNfp):
                    area = abs(pyclipper.Area(finalNfp[j]))
                    if len(finalNfp[j]) < 3 or area < 0.1 * pc_scale * pc_scale:
                        finalNfp.pop(j)
                    else:
                        j += 1

                if not finalNfp or len(finalNfp) == 0:
                    continue

                # Back to normal scale
                f = []
                for j in range(len(finalNfp)):
                    # Convert back to dictionary format
                    poly_dicts = [{'X': pt[0], 'Y': pt[1]} for pt in finalNfp[j]]
                    f.append(toNestCoordinates(poly_dicts, pc_scale))

                finalNfp = f

                # Choose placement that results in the smallest bounding box
                minwidth = None
                minarea = None
                minx = None

                for j in range(len(finalNfp)):
                    nf = finalNfp[j]
                    if abs(GeometryUtil.polygonArea(nf)) < 2:
                        continue

                    for k in range(len(nf)):
                        allpoints = []
                        for m in range(len(placed)):
                            for n in range(len(placed[m])):
                                allpoints.append({
                                    'x': placed[m][n]['x'] + placements[m]['x'],
                                    'y': placed[m][n]['y'] + placements[m]['y']
                                })

                        shiftvector = {
                            'x': nf[k]['x'] - path[0]['x'],
                            'y': nf[k]['y'] - path[0]['y'],
                            'id': path['id'],
                            'rotation': path['rotation'],
                            'nfp': combinedNfp
                        }

                        for m in range(len(path)):
                            allpoints.append({
                                'x': path[m]['x'] + shiftvector['x'],
                                'y': path[m]['y'] + shiftvector['y']
                            })

                        rectbounds = GeometryUtil.getPolygonBounds(allpoints)

                        # Weigh width more, to help compress in direction of gravity
                        area = rectbounds['width'] * 2 + rectbounds['height']

                        if (minarea is None or area < minarea or
                                (GeometryUtil.almostEqual(minarea, area) and (
                                        minx is None or shiftvector['x'] < minx))):
                            minarea = area
                            minwidth = rectbounds['width']
                            position = shiftvector
                            minx = shiftvector['x']

                if position:
                    placed.append(path)
                    placements.append(position)

            if minwidth:
                fitness += minwidth / binarea

            for i in range(len(placed)):
                try:
                    index = paths.index(placed[i])
                    if index >= 0:
                        paths.pop(index)
                except ValueError:
                    pass

            if placements and len(placements) > 0:
                allplacements.append(placements)
            else:
                break  # Something went wrong

        # There were parts that couldn't be placed
        fitness += 2 * len(paths)

        return {
            'placements': allplacements,
            'fitness': fitness,
            'paths': paths,
            'area': binarea
        }