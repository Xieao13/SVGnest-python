#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 * SvgParser
 * A library to convert an SVG string to parse-able segments for CAD/CAM use
 * Licensed under the MIT license
 * Converted from JavaScript to Python
'''

import math
import xml.etree.ElementTree as ET
from lxml import etree
from io import StringIO
import re


class Matrix:
    def __init__(self):
        self.a = 1
        self.b = 0
        self.c = 0
        self.d = 1
        self.e = 0
        self.f = 0

    def matrix(self, params):
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.d = params[3]
        self.e = params[4]
        self.f = params[5]

    def scale(self, sx, sy):
        self.a *= sx
        self.b *= sx
        self.c *= sy
        self.d *= sy

    def rotate(self, angle, cx=0, cy=0):
        angle_rad = angle * math.pi / 180
        cos = math.cos(angle_rad)
        sin = math.sin(angle_rad)

        if cx != 0 or cy != 0:
            self.translate(cx, cy)

        a = self.a
        b = self.b
        c = self.c
        d = self.d

        self.a = a * cos - c * sin
        self.b = b * cos - d * sin
        self.c = a * sin + c * cos
        self.d = b * sin + d * cos

        if cx != 0 or cy != 0:
            self.translate(-cx, -cy)

    def translate(self, tx, ty):
        self.e += tx
        self.f += ty

    def skewX(self, angle):
        angle_rad = angle * math.pi / 180
        self.c += self.a * math.tan(angle_rad)
        self.d += self.b * math.tan(angle_rad)

    def skewY(self, angle):
        angle_rad = angle * math.pi / 180
        self.a += self.c * math.tan(angle_rad)
        self.b += self.d * math.tan(angle_rad)

    def calc(self, x, y):
        return [
            x * self.a + y * self.c + self.e,
            x * self.b + y * self.d + self.f
        ]

    def isIdentity(self):
        return (self.a == 1 and
                self.b == 0 and
                self.c == 0 and
                self.d == 1 and
                self.e == 0 and
                self.f == 0)

    def toArray(self):
        return [self.a, self.b, self.c, self.d, self.e, self.f]


class GeometryUtil:
    @staticmethod
    def almostEqual(a, b, tolerance):
        return abs(a - b) < tolerance

    class QuadraticBezier:
        @staticmethod
        def linearize(start, end, control, tolerance):
            points = []
            points.append(start)

            def recurse(p0, p2, p1, level=0, max_level=10):
                # Check if curve is flat enough
                dx = p2['x'] - p0['x']
                dy = p2['y'] - p0['y']
                d = abs((p1['x'] - p2['x']) * dy - (p1['y'] - p2['y']) * dx)

                if d <= tolerance or level >= max_level:
                    points.append(p2)
                else:
                    # Split curve at midpoint
                    p01 = {'x': (p0['x'] + p1['x']) / 2, 'y': (p0['y'] + p1['y']) / 2}
                    p12 = {'x': (p1['x'] + p2['x']) / 2, 'y': (p1['y'] + p2['y']) / 2}
                    p012 = {'x': (p01['x'] + p12['x']) / 2, 'y': (p01['y'] + p12['y']) / 2}

                    recurse(p0, p012, p01, level + 1, max_level)
                    recurse(p012, p2, p12, level + 1, max_level)

            recurse(start, end, control)
            return points

    class CubicBezier:
        @staticmethod
        def linearize(start, end, control1, control2, tolerance):
            points = []
            points.append(start)

            def recurse(p0, p3, p1, p2, level=0, max_level=10):
                # Check if curve is flat enough
                dx = p3['x'] - p0['x']
                dy = p3['y'] - p0['y']
                d1 = abs((p1['x'] - p3['x']) * dy - (p1['y'] - p3['y']) * dx)
                d2 = abs((p2['x'] - p3['x']) * dy - (p2['y'] - p3['y']) * dx)

                if ((d1 + d2) <= tolerance) or level >= max_level:
                    points.append(p3)
                else:
                    # Split curve at midpoint
                    p01 = {'x': (p0['x'] + p1['x']) / 2, 'y': (p0['y'] + p1['y']) / 2}
                    p12 = {'x': (p1['x'] + p2['x']) / 2, 'y': (p1['y'] + p2['y']) / 2}
                    p23 = {'x': (p2['x'] + p3['x']) / 2, 'y': (p2['y'] + p3['y']) / 2}

                    p012 = {'x': (p01['x'] + p12['x']) / 2, 'y': (p01['y'] + p12['y']) / 2}
                    p123 = {'x': (p12['x'] + p23['x']) / 2, 'y': (p12['y'] + p23['y']) / 2}

                    p0123 = {'x': (p012['x'] + p123['x']) / 2, 'y': (p012['y'] + p123['y']) / 2}

                    recurse(p0, p0123, p01, p012, level + 1, max_level)
                    recurse(p0123, p3, p123, p23, level + 1, max_level)

            recurse(start, end, control1, control2)
            return points

    class Arc:
        @staticmethod
        def linearize(start, end, rx, ry, angle, large_arc_flag, sweep_flag, tolerance):
            points = []
            points.append(start)

            # Convert angle from degrees to radians
            angle_rad = angle * math.pi / 180

            # Get the center parameters
            cx, cy, theta1, theta2 = GeometryUtil.Arc._get_arc_center(
                start['x'], start['y'], end['x'], end['y'],
                large_arc_flag, sweep_flag, rx, ry, angle_rad
            )

            # Calculate how many segments we need
            max_radius = max(rx, ry)
            num_segments = math.ceil(2 * math.pi / math.acos(1 - (tolerance / max_radius)))
            num_segments = max(3, num_segments)

            # Calculate the angle increment
            delta_theta = theta2 - theta1
            if sweep_flag == 0 and delta_theta > 0:
                delta_theta -= 2 * math.pi
            elif sweep_flag == 1 and delta_theta < 0:
                delta_theta += 2 * math.pi

            # Generate points along the arc
            segment_angle = delta_theta / num_segments
            for i in range(1, num_segments + 1):
                theta = theta1 + i * segment_angle

                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)

                cos_angle = math.cos(angle_rad)
                sin_angle = math.sin(angle_rad)

                x = cx + rx * cos_theta * cos_angle - ry * sin_theta * sin_angle
                y = cy + rx * cos_theta * sin_angle + ry * sin_theta * cos_angle

                points.append({'x': x, 'y': y})

            return points

        @staticmethod
        def _get_arc_center(x1, y1, x2, y2, large_arc_flag, sweep_flag, rx, ry, angle_rad):
            # Ensure radii are positive
            rx = abs(rx)
            ry = abs(ry)

            # Step 1: Compute transformed point
            dx = (x1 - x2) / 2
            dy = (y1 - y2) / 2

            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)

            x1p = cos_angle * dx + sin_angle * dy
            y1p = -sin_angle * dx + cos_angle * dy

            # Step 2: Ensure radii are large enough
            rx_sq = rx * rx
            ry_sq = ry * ry
            x1p_sq = x1p * x1p
            y1p_sq = y1p * y1p

            radius_check = x1p_sq / rx_sq + y1p_sq / ry_sq

            if radius_check > 1:
                scaling = math.sqrt(radius_check)
                rx *= scaling
                ry *= scaling
                rx_sq = rx * rx
                ry_sq = ry * ry

            # Step 3: Compute center point
            sign = -1 if large_arc_flag == sweep_flag else 1
            sq = ((rx_sq * ry_sq) - (rx_sq * y1p_sq) - (ry_sq * x1p_sq)) / ((rx_sq * y1p_sq) + (ry_sq * x1p_sq))
            sq = 0 if sq < 0 else sq
            coef = sign * math.sqrt(sq)

            cxp = coef * ((rx * y1p) / ry)
            cyp = coef * (-(ry * x1p) / rx)

            # Step 4: Compute final center point
            cx = cos_angle * cxp - sin_angle * cyp + (x1 + x2) / 2
            cy = sin_angle * cxp + cos_angle * cyp + (y1 + y2) / 2

            # Step 5: Calculate the start and end angles
            ux = (x1p - cxp) / rx
            uy = (y1p - cyp) / ry
            vx = (-x1p - cxp) / rx
            vy = (-y1p - cyp) / ry

            # Calculate the start angle
            theta1 = GeometryUtil.Arc._angle_between(1, 0, ux, uy)

            # Calculate the delta angle
            delta_theta = GeometryUtil.Arc._angle_between(ux, uy, vx, vy)

            if sweep_flag == 0 and delta_theta > 0:
                delta_theta -= 2 * math.pi
            elif sweep_flag == 1 and delta_theta < 0:
                delta_theta += 2 * math.pi

            # Calculate end angle
            theta2 = theta1 + delta_theta

            return cx, cy, theta1, theta2

        @staticmethod
        def _angle_between(ux, uy, vx, vy):
            # Calculate angle between two vectors
            dot = ux * vx + uy * vy
            len_u = math.sqrt(ux * ux + uy * uy)
            len_v = math.sqrt(vx * vx + vy * vy)

            # Avoid division by zero
            if len_u * len_v == 0:
                return 0

            cos_angle = max(-1, min(1, dot / (len_u * len_v)))
            angle = math.acos(cos_angle)

            # Determine the sign
            cross = ux * vy - uy * vx
            if cross < 0:
                angle = -angle

            return angle


class SvgParser:
    def __init__(self):
        # the SVG document
        self.svg = None

        # the top level SVG element of the SVG document
        self.svgRoot = None

        self.allowedElements = ['svg', 'circle', 'ellipse', 'path', 'polygon', 'polyline', 'rect', 'line']

        self.conf = {
            'tolerance': 2,  # max bound for bezier->line segment conversion, in native SVG units
            'toleranceSvg': 0.005  # fudge factor for browser inaccuracy in SVG unit handling
        }

    def config(self, config):
        if 'tolerance' in config:
            self.conf['tolerance'] = config['tolerance']
        return self

    def load(self, svgString):
        if not svgString or not isinstance(svgString, str):
            raise ValueError('invalid SVG string')

        parser = etree.XMLParser(remove_blank_text=True)
        svg = etree.parse(StringIO(svgString), parser)

        self.svgRoot = False

        if svg:
            self.svg = svg
            root = svg.getroot()

            if root.tag.endswith('svg'):
                self.svgRoot = root
        else:
            raise ValueError("Failed to parse SVG string")

        if not self.svgRoot:
            raise ValueError("SVG has no children")

        return self.svgRoot

    def cleanInput(self):
        # apply any transformations
        self.applyTransform(self.svgRoot)

        # remove any g elements and bring all elements to the top level
        self.flatten(self.svgRoot)

        # remove any non-contour elements like text
        self.filter(self.allowedElements)

        # split any compound paths into individual path elements
        self.recurse(self.svgRoot, self.splitPath)

        return self.svgRoot

    def getStyle(self):
        if not self.svgRoot:
            return False

        for child in self.svgRoot:
            if child.tag.endswith('style'):
                return child

        return False

    def _parse_path_data(self, d):
        """Parse SVG path data string into segments"""
        commands = []
        current_cmd = None
        current_params = []
        i = 0

        # Regular expressions for numbers and commands
        cmd_regex = re.compile(r'[MmLlHhVvCcSsQqTtAaZz]')
        num_regex = re.compile(r'-?(?:\d*\.\d+|\d+)')

        while i < len(d):
            if d[i].isspace() or d[i] == ',':
                i += 1
                continue

            # Check for a command
            if cmd_regex.match(d[i]):
                if current_cmd is not None:
                    commands.append([current_cmd] + current_params)

                current_cmd = d[i]
                current_params = []
                i += 1

            # Check for a number
            else:
                num_match = num_regex.match(d[i:])
                if num_match:
                    num_str = num_match.group()
                    current_params.append(float(num_str))
                    i += len(num_str)
                else:
                    # Skip any unexpected characters
                    i += 1

        # Add the last command
        if current_cmd is not None:
            commands.append([current_cmd] + current_params)

        return commands

    def _format_path_segments(self, segments):
        """Format segments back into SVG path data"""
        d = []

        for segment in segments:
            cmd = segment[0]
            params = segment[1:]

            if cmd == 'Z' or cmd == 'z':
                d.append(cmd)
            else:
                param_str = ' '.join([str(round(p, 6)) for p in params])
                d.append(f"{cmd} {param_str}")

        return ' '.join(d)

    def pathToAbsolute(self, path):
        if not path or not path.tag.endswith('path'):
            raise ValueError('invalid path')

        d = path.get('d', '')
        if not d:
            return

        segments = self._parse_path_data(d)

        # Convert to absolute coordinates
        x, y = 0, 0
        x0, y0 = 0, 0
        x1, y1 = 0, 0
        x2, y2 = 0, 0

        absolute_segments = []

        for i, segment in enumerate(segments):
            cmd = segment[0]
            params = segment[1:]

            # Uppercase commands are already absolute
            if cmd.isupper():
                if cmd == 'M':
                    x, y = params[0], params[1]
                    x0, y0 = x, y
                elif cmd == 'L':
                    x, y = params[0], params[1]
                elif cmd == 'H':
                    x = params[0]
                elif cmd == 'V':
                    y = params[0]
                elif cmd == 'C':
                    x1, y1 = params[0], params[1]
                    x2, y2 = params[2], params[3]
                    x, y = params[4], params[5]
                elif cmd == 'S':
                    x2, y2 = params[0], params[1]
                    x, y = params[2], params[3]
                elif cmd == 'Q':
                    x1, y1 = params[0], params[1]
                    x, y = params[2], params[3]
                elif cmd == 'T':
                    x, y = params[0], params[1]
                elif cmd == 'A':
                    x, y = params[5], params[6]

                absolute_segments.append(segment)
            else:
                # Convert relative commands to absolute
                if cmd == 'm':
                    # First m is treated as absolute M
                    if i == 0:
                        x += params[0]
                        y += params[1]
                        x0, y0 = x, y
                        absolute_segments.append(['M', x, y])
                    else:
                        x += params[0]
                        y += params[1]
                        x0, y0 = x, y
                        absolute_segments.append(['M', x, y])
                elif cmd == 'l':
                    x += params[0]
                    y += params[1]
                    absolute_segments.append(['L', x, y])
                elif cmd == 'h':
                    x += params[0]
                    absolute_segments.append(['L', x, y])
                elif cmd == 'v':
                    y += params[0]
                    absolute_segments.append(['L', x, y])
                elif cmd == 'c':
                    x1 = x + params[0]
                    y1 = y + params[1]
                    x2 = x + params[2]
                    y2 = y + params[3]
                    x += params[4]
                    y += params[5]
                    absolute_segments.append(['C', x1, y1, x2, y2, x, y])
                elif cmd == 's':
                    # If previous command was a C or S, the first control point is reflection of the previous control point
                    if absolute_segments and absolute_segments[-1][0] in 'CS':
                        prev = absolute_segments[-1]
                        x1 = x - (prev[-4] - x)
                        y1 = y - (prev[-3] - y)
                    else:
                        x1, y1 = x, y

                    x2 = x + params[0]
                    y2 = y + params[1]
                    x += params[2]
                    y += params[3]
                    absolute_segments.append(['C', x1, y1, x2, y2, x, y])
                elif cmd == 'q':
                    x1 = x + params[0]
                    y1 = y + params[1]
                    x += params[2]
                    y += params[3]
                    absolute_segments.append(['Q', x1, y1, x, y])
                elif cmd == 't':
                    # If previous command was a Q or T, the control point is reflection of the previous control point
                    if absolute_segments and absolute_segments[-1][0] in 'QT':
                        prev = absolute_segments[-1]
                        x1 = x - (prev[1] - x)
                        y1 = y - (prev[2] - y)
                    else:
                        x1, y1 = x, y

                    x += params[0]
                    y += params[1]
                    absolute_segments.append(['Q', x1, y1, x, y])
                elif cmd == 'a':
                    rx, ry, angle, large_arc, sweep, dx, dy = params
                    x += dx
                    y += dy
                    absolute_segments.append(['A', rx, ry, angle, large_arc, sweep, x, y])
                elif cmd == 'z':
                    x, y = x0, y0
                    absolute_segments.append(['Z'])

            # Record the start of a subpath
            if cmd.upper() == 'M':
                x0, y0 = x, y

        # Update the path data
        path.set('d', self._format_path_segments(absolute_segments))

    def transformParse(self, transformString):
        if not transformString:
            return Matrix()

        matrix = Matrix()

        # Parse the transform string
        transforms = re.findall(r'(\w+)\s*\(([^)]*)\)', transformString)

        for cmd, params_str in transforms:
            params = [float(p) for p in re.split(r'[\s,]+', params_str.strip()) if p]

            if cmd == 'matrix' and len(params) == 6:
                matrix.matrix(params)
            elif cmd == 'translate':
                if len(params) == 1:
                    matrix.translate(params[0], 0)
                elif len(params) >= 2:
                    matrix.translate(params[0], params[1])
            elif cmd == 'scale':
                if len(params) == 1:
                    matrix.scale(params[0], params[0])
                elif len(params) >= 2:
                    matrix.scale(params[0], params[1])
            elif cmd == 'rotate':
                if len(params) == 1:
                    matrix.rotate(params[0], 0, 0)
                elif len(params) >= 3:
                    matrix.rotate(params[0], params[1], params[2])
            elif cmd == 'skewX' and len(params) >= 1:
                matrix.skewX(params[0])
            elif cmd == 'skewY' and len(params) >= 1:
                matrix.skewY(params[0])

        return matrix

    def applyTransform(self, element, globalTransform=''):
        if element is None:
            return

        transformString = element.get('transform', '')
        transformString = globalTransform + transformString

        transform = None
        if transformString:
            transform = self.transformParse(transformString)

        if not transform:
            transform = Matrix()

        tarray = transform.toArray()

        # decompose affine matrix to rotate, scale components
        rotate = math.atan2(tarray[1], tarray[3]) * 180 / math.pi
        scale = math.sqrt(tarray[0] * tarray[0] + tarray[2] * tarray[2])

        tag = element.tag.split('}')[-1]

        if tag in ['g', 'svg', 'defs', 'clipPath']:
            if 'transform' in element.attrib:
                del element.attrib['transform']

            for child in list(element):
                self.applyTransform(child, transformString)

        elif transform and not transform.isIdentity():
            element_id = element.get('id')
            element_class = element.get('class')

            if tag == 'ellipse':
                # Replace ellipse with path
                path = etree.Element('{http://www.w3.org/2000/svg}path')

                cx = float(element.get('cx', 0))
                cy = float(element.get('cy', 0))
                rx = float(element.get('rx', 0))
                ry = float(element.get('ry', 0))

                d = f"M {cx - rx},{cy} A {rx},{ry} 0 1 0 {cx + rx},{cy} A {rx},{ry} 0 1 0 {cx - rx},{cy} Z"
                path.set('d', d)

                if transformString:
                    path.set('transform', transformString)

                # Replace the element with the path
                parent = element.getparent()
                if parent is not None:
                    parent.replace(element, path)

                element = path
                tag = 'path'

            if tag == 'path':
                self.pathToAbsolute(element)

                d = element.get('d', '')
                if not d:
                    return

                segments = self._parse_path_data(d)

                prevx, prevy = 0, 0
                transformed_segments = []

                for segment in segments:
                    cmd = segment[0]
                    params = segment[1:]

                    if cmd == 'H':
                        # Convert H to L
                        x = params[0]
                        transformed_segments.append(['L', x, prevy])
                        prevx, prevy = x, prevy
                    elif cmd == 'V':
                        # Convert V to L
                        y = params[0]
                        transformed_segments.append(['L', prevx, y])
                        prevx, prevy = prevx, y
                    elif cmd == 'A':
                        # Transform arc
                        rx, ry, angle, large_arc, sweep, x, y = params

                        # Apply scale to radius
                        rx *= scale
                        ry *= scale

                        # Apply rotation to angle
                        angle += rotate

                        # Transform end point
                        tx, ty = transform.calc(x, y)

                        transformed_segments.append(['A', rx, ry, angle, large_arc, sweep, tx, ty])
                        prevx, prevy = x, y
                    elif cmd in 'ML':
                        # Transform point
                        x, y = params
                        tx, ty = transform.calc(x, y)

                        transformed_segments.append([cmd, tx, ty])
                        prevx, prevy = x, y
                    elif cmd == 'C':
                        # Transform cubic bezier
                        x1, y1, x2, y2, x, y = params

                        tx1, ty1 = transform.calc(x1, y1)
                        tx2, ty2 = transform.calc(x2, y2)
                        tx, ty = transform.calc(x, y)

                        transformed_segments.append([cmd, tx1, ty1, tx2, ty2, tx, ty])
                        prevx, prevy = x, y
                    elif cmd == 'S':
                        # Transform smooth cubic bezier
                        x2, y2, x, y = params

                        tx2, ty2 = transform.calc(x2, y2)
                        tx, ty = transform.calc(x, y)

                        transformed_segments.append([cmd, tx2, ty2, tx, ty])
                        prevx, prevy = x, y
                    elif cmd == 'Q':
                        # Transform quadratic bezier
                        x1, y1, x, y = params

                        tx1, ty1 = transform.calc(x1, y1)
                        tx, ty = transform.calc(x, y)

                        transformed_segments.append([cmd, tx1, ty1, tx, ty])
                        prevx, prevy = x, y
                    elif cmd == 'T':
                        # Transform smooth quadratic bezier
                        x, y = params

                        tx, ty = transform.calc(x, y)

                        transformed_segments.append([cmd, tx, ty])
                        prevx, prevy = x, y
                    elif cmd in 'Zz':
                        transformed_segments.append([cmd])

                # Update path data
                element.set('d', self._format_path_segments(transformed_segments))
                if 'transform' in element.attrib:
                    del element.attrib['transform']

            elif tag == 'circle':
                # Transform circle
                cx = float(element.get('cx', 0))
                cy = float(element.get('cy', 0))
                r = float(element.get('r', 0))

                # Transform center
                tx, ty = transform.calc(cx, cy)

                element.set('cx', str(tx))
                element.set('cy', str(ty))

                # Apply scale to radius
                element.set('r', str(r * scale))

                if 'transform' in element.attrib:
                    del element.attrib['transform']

            elif tag == 'line':
                # Transform line
                x1 = float(element.get('x1', 0))
                y1 = float(element.get('y1', 0))
                x2 = float(element.get('x2', 0))
                y2 = float(element.get('y2', 0))

                # Transform points
                tx1, ty1 = transform.calc(x1, y1)
                tx2, ty2 = transform.calc(x2, y2)

                element.set('x1', str(tx1))
                element.set('y1', str(ty1))
                element.set('x2', str(tx2))
                element.set('y2', str(ty2))

                if 'transform' in element.attrib:
                    del element.attrib['transform']

            elif tag == 'rect':
                # Replace rect with polygon
                polygon = etree.Element('{http://www.w3.org/2000/svg}polygon')

                x = float(element.get('x', 0))
                y = float(element.get('y', 0))
                width = float(element.get('width', 0))
                height = float(element.get('height', 0))

                # Create points for the rectangle
                points = f"{x},{y} {x + width},{y} {x + width},{y + height} {x},{y + height}"
                polygon.set('points', points)

                if 'transform' in element.attrib:
                    polygon.set('transform', element.get('transform'))

                # Replace the element with the polygon
                parent = element.getparent()
                if parent is not None:
                    parent.replace(element, polygon)

                element = polygon
                tag = 'polygon'

            if tag in ['polygon', 'polyline']:
                # Transform polygon or polyline
                points_str = element.get('points', '')
                if not points_str:
                    return

                # Parse points
                points = []
                for point_str in points_str.split():
                    if ',' in point_str:
                        x_str, y_str = point_str.split(',')
                        try:
                            x = float(x_str)
                            y = float(y_str)
                            points.append((x, y))
                        except ValueError:
                            continue

                # Transform points
                transformed_points = []
                for x, y in points:
                    tx, ty = transform.calc(x, y)
                    transformed_points.append(f"{tx},{ty}")

                element.set('points', ' '.join(transformed_points))

                if 'transform' in element.attrib:
                    del element.attrib['transform']

            # Restore ID and class attributes if they exist
            if element_id:
                element.set('id', element_id)
            if element_class:
                element.set('class', element_class)

    def flatten(self, element):
        """Bring all child elements to the top level"""
        if element is None:
            return

        # Process children first (depth-first)
        children = list(element)
        for child in children:
            self.flatten(child)

        # Skip the root svg element
        tag = element.tag.split('}')[-1]
        if tag != 'svg':
            parent = element.getparent()
            if parent is not None:
                # Move all children to the parent
                for child in list(element):
                    index = parent.index(element)
                    parent.insert(index, child)

    def filter(self, whitelist, element=None):
        """Remove all elements with tag name not in the whitelist"""
        if not whitelist or len(whitelist) == 0:
            raise ValueError('invalid whitelist')

        element = element or self.svgRoot
        if element is None:
            return

        # Process children first (depth-first)
        children = list(element)
        for child in children:
            self.filter(whitelist, child)

        # Check if this element should be removed
        tag = element.tag.split('}')[-1]
        if len(list(element)) == 0 and tag not in whitelist:
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)

    def splitPath(self, path):
        """Split a compound path (paths with multiple M commands) into separate paths"""
        if path is None or path.tag.split('}')[-1] != 'path' or path.getparent() is None:
            return False

        # Get the path data
        d = path.get('d', '')
        if not d:
            return False

        # Parse the path data
        segments = self._parse_path_data(d)

        # Count M commands
        m_count = sum(1 for segment in segments if segment[0] in 'Mm')

        if m_count <= 1:
            return False  # Only 1 M command, no need to split

        # Split into separate paths
        paths = []
        current_path_segments = []

        for segment in segments:
            cmd = segment[0]

            if cmd in 'Mm' and current_path_segments:
                # Start a new path
                new_path = etree.Element(path.tag)
                for key, value in path.attrib.items():
                    if key != 'd':
                        new_path.set(key, value)

                new_path.set('d', self._format_path_segments(current_path_segments))
                paths.append(new_path)
                current_path_segments = []

            current_path_segments.append(segment)

        # Add the last path
        if current_path_segments:
            new_path = etree.Element(path.tag)
            for key, value in path.attrib.items():
                if key != 'd':
                    new_path.set(key, value)

            new_path.set('d', self._format_path_segments(current_path_segments))
            paths.append(new_path)

        # Add all paths to the parent
        parent = path.getparent()
        added_paths = []

        for new_path in paths:
            # Don't add trivial paths from sequential M commands
            if len(self._parse_path_data(new_path.get('d', ''))) > 1:
                index = parent.index(path)
                parent.insert(index, new_path)
                added_paths.append(new_path)

        # Remove the original path
        parent.remove(path)

        return added_paths if added_paths else False

    def recurse(self, element, func):
        """Recursively run the given function on the given element"""
        if element is None:
            return

        # Make a copy of the children to avoid modification issues during iteration
        children = list(element)
        for child in children:
            self.recurse(child, func)

        func(element)

    def polygonify(self, element):
        """Return a polygon from the given SVG element in the form of an array of points"""
        if element is None:
            return []

        tag = element.tag.split('}')[-1]
        poly = []

        if tag in ['polygon', 'polyline']:
            points_str = element.get('points', '')
            if not points_str:
                return poly

            for point_str in points_str.split():
                if ',' in point_str:
                    x_str, y_str = point_str.split(',')
                    try:
                        x = float(x_str)
                        y = float(y_str)
                        poly.append({'x': x, 'y': y})
                    except ValueError:
                        pass  # Skip invalid points

        elif tag == 'rect':
            x = float(element.get('x', 0))
            y = float(element.get('y', 0))
            width = float(element.get('width', 0))
            height = float(element.get('height', 0))

            poly.append({'x': x, 'y': y})
            poly.append({'x': x + width, 'y': y})
            poly.append({'x': x + width, 'y': y + height})
            poly.append({'x': x, 'y': y + height})

        elif tag == 'circle':
            radius = float(element.get('r', 0))
            cx = float(element.get('cx', 0))
            cy = float(element.get('cy', 0))

            # Calculate the number of segments needed to approximate the circle
            num = math.ceil((2 * math.pi) / math.acos(1 - (self.conf['tolerance'] / radius)))
            num = max(3, num)

            for i in range(num):
                theta = i * (2 * math.pi / num)
                x = radius * math.cos(theta) + cx
                y = radius * math.sin(theta) + cy
                poly.append({'x': x, 'y': y})

        elif tag == 'ellipse':
            rx = float(element.get('rx', 0))
            ry = float(element.get('ry', 0))
            cx = float(element.get('cx', 0))
            cy = float(element.get('cy', 0))

            max_radius = max(rx, ry)
            num = math.ceil((2 * math.pi) / math.acos(1 - (self.conf['tolerance'] / max_radius)))
            num = max(3, num)

            for i in range(num):
                theta = i * (2 * math.pi / num)
                x = rx * math.cos(theta) + cx
                y = ry * math.sin(theta) + cy
                poly.append({'x': x, 'y': y})

        elif tag == 'path':
            # Get the path data
            d = element.get('d', '')
            if not d:
                return poly

            # Make sure path uses absolute coordinates
            self.pathToAbsolute(element)
            d = element.get('d', '')

            # Parse the path data
            segments = self._parse_path_data(d)

            # Process path segments
            x, y = 0, 0
            x0, y0 = 0, 0  # Starting point of current subpath
            x1, y1 = 0, 0  # Control points
            x2, y2 = 0, 0
            prevx, prevy = 0, 0
            prevx1, prevy1 = 0, 0
            prevx2, prevy2 = 0, 0

            for segment in segments:
                cmd = segment[0]
                params = segment[1:]

                prevx, prevy = x, y
                prevx1, prevy1 = x1, y1
                prevx2, prevy2 = x2, y2

                if cmd in 'MLHVCSQTA':
                    if cmd == 'H':
                        x = params[0]
                    elif cmd == 'V':
                        y = params[0]
                    elif cmd == 'M' or cmd == 'L':
                        x, y = params[0], params[1]
                    elif cmd == 'C':
                        x1, y1 = params[0], params[1]
                        x2, y2 = params[2], params[3]
                        x, y = params[4], params[5]
                    elif cmd == 'S':
                        x2, y2 = params[0], params[1]
                        x, y = params[2], params[3]
                        # If previous command was a C or S, first control point is reflection of previous second control point
                        if segments and segments[-1][0] in 'CS':
                            x1 = prevx + (prevx - prevx2)
                            y1 = prevy + (prevy - prevy2)
                        else:
                            x1, y1 = prevx, prevy
                    elif cmd == 'Q':
                        x1, y1 = params[0], params[1]
                        x, y = params[2], params[3]
                    elif cmd == 'T':
                        x, y = params[0], params[1]
                        # If previous command was a Q or T, control point is reflection of previous control point
                        if segments and segments[-1][0] in 'QT':
                            x1 = prevx + (prevx - prevx1)
                            y1 = prevy + (prevy - prevy1)
                        else:
                            x1, y1 = prevx, prevy
                    elif cmd == 'A' and len(params) >= 7:
                        x, y = params[5], params[6]

                # Process different commands
                if cmd in 'ML':
                    # Linear segments - add point directly
                    poly.append({'x': x, 'y': y})

                elif cmd == 'H':
                    poly.append({'x': x, 'y': y})

                elif cmd == 'V':
                    poly.append({'x': x, 'y': y})

                elif cmd == 'Q' or cmd == 'T':
                    # Quadratic Bezier curves - linearize
                    start = {'x': prevx, 'y': prevy}
                    end = {'x': x, 'y': y}
                    control = {'x': x1, 'y': y1}

                    points = GeometryUtil.QuadraticBezier.linearize(start, end, control, self.conf['tolerance'])
                    # Skip first point as it would already be in poly
                    for i in range(1, len(points)):
                        poly.append(points[i])

                elif cmd == 'C' or cmd == 'S':
                    # Cubic Bezier curves - linearize
                    start = {'x': prevx, 'y': prevy}
                    end = {'x': x, 'y': y}
                    control1 = {'x': x1, 'y': y1}
                    control2 = {'x': x2, 'y': y2}

                    points = GeometryUtil.CubicBezier.linearize(start, end, control1, control2, self.conf['tolerance'])
                    # Skip first point as it would already be in poly
                    for i in range(1, len(points)):
                        poly.append(points[i])

                elif cmd == 'A':
                    # Arc - linearize
                    start = {'x': prevx, 'y': prevy}
                    end = {'x': x, 'y': y}
                    if len(params) >= 5:
                        rx, ry, angle = params[0:3]
                        large_arc_flag = int(params[3])
                        sweep_flag = int(params[4])

                        points = GeometryUtil.Arc.linearize(
                            start, end, rx, ry, angle, large_arc_flag, sweep_flag, self.conf['tolerance']
                        )
                        # Skip first point as it would already be in poly
                        for i in range(1, len(points)):
                            poly.append(points[i])

                # Record the start of a subpath
                if cmd == 'M':
                    x0, y0 = x, y

                # Close path
                if cmd == 'Z':
                    x, y = x0, y0

        # Do not include last point if coincident with starting point
        while (len(poly) > 1 and
               GeometryUtil.almostEqual(poly[0]['x'], poly[-1]['x'], self.conf['toleranceSvg']) and
               GeometryUtil.almostEqual(poly[0]['y'], poly[-1]['y'], self.conf['toleranceSvg'])):
            poly.pop()

        return poly


# Create a global SvgParser instance and expose public methods
parser = SvgParser()


# Public API
def config(config_obj):
    return parser.config(config_obj)


def load(svg_string):
    return parser.load(svg_string)


def getStyle():
    return parser.getStyle()


def clean():
    return parser.cleanInput()


def polygonify(element):
    return parser.polygonify(element)


# For module compatibility with original JS version
SvgParser = {
    'config': config,
    'load': load,
    'getStyle': getStyle,
    'clean': clean,
    'polygonify': polygonify
}