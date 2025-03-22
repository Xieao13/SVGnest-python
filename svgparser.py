#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SvgParser
A library to convert an SVG string to parse-able segments for CAD/CAM use
Converted from JavaScript to Python
"""

import xml.etree.ElementTree as ET
import math
import numpy as np
from lxml import etree
from io import StringIO


class Matrix:
    """
    Helper class for transformation matrix operations
    Equivalent to the Matrix class used in the JS version
    """

    def __init__(self):
        self.a = 1
        self.b = 0
        self.c = 0
        self.d = 1
        self.e = 0
        self.f = 0

    def matrix(self, params):
        # Apply matrix transformation
        a, b, c, d, e, f = params
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def translate(self, tx, ty):
        # Apply translation
        self.e += tx
        self.f += ty

    def scale(self, sx, sy):
        # Apply scaling
        self.a *= sx
        self.b *= sx
        self.c *= sy
        self.d *= sy

    def rotate(self, angle, cx=0, cy=0):
        # Apply rotation
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

    def skewX(self, angle):
        # Apply skew in X direction
        angle_rad = angle * math.pi / 180
        self.c += self.a * math.tan(angle_rad)
        self.d += self.b * math.tan(angle_rad)

    def skewY(self, angle):
        # Apply skew in Y direction
        angle_rad = angle * math.pi / 180
        self.a += self.c * math.tan(angle_rad)
        self.b += self.d * math.tan(angle_rad)

    def calc(self, x, y):
        # Calculate transformed coordinates
        return [
            x * self.a + y * self.c + self.e,
            x * self.b + y * self.d + self.f
        ]

    def isIdentity(self):
        # Check if this is an identity matrix
        return (self.a == 1 and
                self.b == 0 and
                self.c == 0 and
                self.d == 1 and
                self.e == 0 and
                self.f == 0)

    def toArray(self):
        # Convert to array format
        return [self.a, self.b, self.c, self.d, self.e, self.f]


class GeometryUtil:
    """
    Utility class for geometric operations and calculations
    """

    @staticmethod
    def almostEqual(a, b, tolerance):
        return abs(a - b) < tolerance

    class QuadraticBezier:
        @staticmethod
        def linearize(p0, p2, p1, tolerance):
            """
            Convert a quadratic bezier curve to a series of points
            p0: start point, p2: end point, p1: control point
            """
            points = []

            # Use De Casteljau's algorithm to subdivide the curve until it's flat enough
            def flatten_recursive(p0, p2, p1, level=0, max_level=10):
                # Check if the curve is flat enough
                d = abs((p0['x'] - p2['x']) * (p1['y'] - p0['y']) - (p0['x'] - p1['x']) * (p2['y'] - p0['y']))
                if d <= tolerance or level >= max_level:
                    points.append(p0)
                    points.append(p2)
                else:
                    # Split the curve
                    p01 = {'x': (p0['x'] + p1['x']) / 2, 'y': (p0['y'] + p1['y']) / 2}
                    p12 = {'x': (p1['x'] + p2['x']) / 2, 'y': (p1['y'] + p2['y']) / 2}
                    p012 = {'x': (p01['x'] + p12['x']) / 2, 'y': (p01['y'] + p12['y']) / 2}

                    # Recurse on the two halves
                    flatten_recursive(p0, p012, p01, level + 1, max_level)
                    flatten_recursive(p012, p2, p12, level + 1, max_level)

            flatten_recursive(p0, p2, p1)
            # Remove duplicates and sort points
            return points

    class CubicBezier:
        @staticmethod
        def linearize(p0, p3, p1, p2, tolerance):
            """
            Convert a cubic bezier curve to a series of points
            p0: start point, p3: end point, p1, p2: control points
            """
            points = []

            def flatten_recursive(p0, p3, p1, p2, level=0, max_level=10):
                # Check if the curve is flat enough
                d1 = max(abs(p0['x'] - p1['x']), abs(p0['y'] - p1['y']))
                d2 = max(abs(p1['x'] - p2['x']), abs(p1['y'] - p2['y']))
                d3 = max(abs(p2['x'] - p3['x']), abs(p2['y'] - p3['y']))

                if (d1 + d2 + d3) <= tolerance or level >= max_level:
                    points.append(p0)
                    points.append(p3)
                else:
                    # Split the curve at t=0.5
                    p01 = {'x': (p0['x'] + p1['x']) / 2, 'y': (p0['y'] + p1['y']) / 2}
                    p12 = {'x': (p1['x'] + p2['x']) / 2, 'y': (p1['y'] + p2['y']) / 2}
                    p23 = {'x': (p2['x'] + p3['x']) / 2, 'y': (p2['y'] + p3['y']) / 2}

                    p012 = {'x': (p01['x'] + p12['x']) / 2, 'y': (p01['y'] + p12['y']) / 2}
                    p123 = {'x': (p12['x'] + p23['x']) / 2, 'y': (p12['y'] + p23['y']) / 2}

                    p0123 = {'x': (p012['x'] + p123['x']) / 2, 'y': (p012['y'] + p123['y']) / 2}

                    # Recurse on the two halves
                    flatten_recursive(p0, p0123, p01, p012, level + 1, max_level)
                    flatten_recursive(p0123, p3, p123, p23, level + 1, max_level)

            flatten_recursive(p0, p3, p1, p2)
            return points

    class Arc:
        @staticmethod
        def linearize(p0, p1, rx, ry, angle, large_arc_flag, sweep_flag, tolerance):
            """
            Convert an SVG arc to a series of points
            """
            points = [p0]  # Start with the first point

            # Convert angle from degrees to radians
            angle_rad = angle * math.pi / 180

            # Get the center parameters
            cx, cy, theta1, theta2 = GeometryUtil.Arc._get_arc_center(
                p0['x'], p0['y'], p1['x'], p1['y'], large_arc_flag, sweep_flag, rx, ry, angle_rad
            )

            # Calculate how many segments we need
            max_radius = max(rx, ry)
            num_segments = math.ceil((2 * math.pi) / math.acos(1 - (tolerance / max_radius)))

            # Ensure at least 3 segments
            num_segments = max(3, num_segments)

            # Calculate the angle increment
            delta_theta = theta2 - theta1
            if sweep_flag == 0 and delta_theta > 0:
                delta_theta -= 2 * math.pi
            elif sweep_flag == 1 and delta_theta < 0:
                delta_theta += 2 * math.pi

            # Determine segment angle
            segment_angle = delta_theta / num_segments

            # Generate points along the arc
            for i in range(1, num_segments + 1):
                theta = theta1 + i * segment_angle

                # Calculate point on the arc
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)

                x = cx + rx * cos_theta * math.cos(angle_rad) - ry * sin_theta * math.sin(angle_rad)
                y = cy + rx * cos_theta * math.sin(angle_rad) + ry * sin_theta * math.cos(angle_rad)

                points.append({'x': x, 'y': y})

            return points

        @staticmethod
        def _get_arc_center(x1, y1, x2, y2, large_arc_flag, sweep_flag, rx, ry, angle_rad):
            """
            Get the center point of an elliptical arc
            F.6.5 of the SVG specification 1.1
            """
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
                rx *= math.sqrt(radius_check)
                ry *= math.sqrt(radius_check)
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
            """Calculate angle between two vectors"""
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
    """
    Main class for parsing SVG files
    """

    def __init__(self):
        # the SVG document
        self.svg = None

        # the top level SVG element of the SVG document
        self.svg_root = None

        self.allowed_elements = ['svg', 'circle', 'ellipse', 'path', 'polygon', 'polyline', 'rect', 'line']

        self.conf = {
            'tolerance': 2,  # max bound for bezier->line segment conversion, in native SVG units
            'toleranceSvg': 0.005  # fudge factor for browser inaccuracy in SVG unit handling
        }

    def config(self, config):
        """Update configuration parameters"""
        if 'tolerance' in config:
            self.conf['tolerance'] = config['tolerance']
        return self

    def load(self, svg_string):
        """Load and parse an SVG string"""
        if not svg_string or not isinstance(svg_string, str):
            raise ValueError('Invalid SVG string')

        try:
            # Parse the SVG with lxml
            parser = etree.XMLParser(remove_blank_text=True)
            svg = etree.parse(StringIO(svg_string), parser)
            self.svg = svg

            # Get the root element
            self.svg_root = svg.getroot()

            if self.svg_root is None or self.svg_root.tag != '{http://www.w3.org/2000/svg}svg':
                raise ValueError('SVG has no root element')

            return self.svg_root
        except Exception as e:
            raise ValueError(f"Failed to parse SVG string: {str(e)}")

    def clean_input(self):
        """Prepare the SVG for CAD-CAM/nest related operations"""

        # Apply any transformations
        self.apply_transform(self.svg_root)

        # Remove any g elements and bring all elements to the top level
        self.flatten(self.svg_root)

        # Remove any non-contour elements like text
        self.filter(self.allowed_elements)

        # Split any compound paths into individual path elements
        self.recurse(self.svg_root, self.split_path)

        return self.svg_root

    def get_style(self):
        """Return style node, if any"""
        if not self.svg_root:
            return False

        for child in self.svg_root:
            if child.tag.endswith('style'):
                return child

        return False

    def path_to_absolute(self, path):
        """Convert path coordinates to absolute values"""
        if path is None or not path.tag.endswith('path'):
            raise ValueError('Invalid path')

        # Get the path data
        d = path.get('d', '')
        if not d:
            return

        # Parse the path data
        segments = self._parse_path(d)

        # Convert to absolute coordinates
        x, y = 0, 0
        x0, y0 = 0, 0
        x1, y1 = 0, 0
        x2, y2 = 0, 0

        absolute_segments = []

        for segment in segments:
            command = segment[0]
            params = segment[1:]

            # Uppercase commands are already absolute
            if command.upper() == command:
                if command in 'MLHVCSQTA':
                    if command == 'H':
                        x = params[0]
                    elif command == 'V':
                        y = params[0]
                    elif command in 'MLCSQT':
                        x = params[-2]
                        y = params[-1]
                    elif command == 'A':
                        x = params[5]
                        y = params[6]
                absolute_segments.append([command] + params)
            else:
                # Convert relative commands to absolute
                if command == 'm':
                    # First m is treated as M
                    if len(absolute_segments) == 0:
                        x += params[0]
                        y += params[1]
                        absolute_segments.append(['M', x, y])
                    else:
                        x += params[0]
                        y += params[1]
                        absolute_segments.append(['M', x, y])
                elif command == 'l':
                    x += params[0]
                    y += params[1]
                    absolute_segments.append(['L', x, y])
                elif command == 'h':
                    x += params[0]
                    absolute_segments.append(['L', x, y])
                elif command == 'v':
                    y += params[0]
                    absolute_segments.append(['L', x, y])
                elif command == 'c':
                    x1 = x + params[0]
                    y1 = y + params[1]
                    x2 = x + params[2]
                    y2 = y + params[3]
                    x += params[4]
                    y += params[5]
                    absolute_segments.append(['C', x1, y1, x2, y2, x, y])
                elif command == 's':
                    # If previous command was a C or S, the first control point is reflection of the previous control point
                    if absolute_segments and absolute_segments[-1][0] in 'CS':
                        x1 = x - (absolute_segments[-1][3] - x)
                        y1 = y - (absolute_segments[-1][4] - y)
                    else:
                        x1 = x
                        y1 = y
                    x2 = x + params[0]
                    y2 = y + params[1]
                    x += params[2]
                    y += params[3]
                    absolute_segments.append(['C', x1, y1, x2, y2, x, y])
                elif command == 'q':
                    x1 = x + params[0]
                    y1 = y + params[1]
                    x += params[2]
                    y += params[3]
                    absolute_segments.append(['Q', x1, y1, x, y])
                elif command == 't':
                    # If previous command was a Q or T, the control point is reflection of the previous control point
                    if absolute_segments and absolute_segments[-1][0] in 'QT':
                        x1 = x - (absolute_segments[-1][1] - x)
                        y1 = y - (absolute_segments[-1][2] - y)
                    else:
                        x1 = x
                        y1 = y
                    x += params[0]
                    y += params[1]
                    absolute_segments.append(['Q', x1, y1, x, y])
                elif command == 'a':
                    # A rx ry x-axis-rotation large-arc-flag sweep-flag x y
                    rx, ry, angle, large_arc, sweep, dx, dy = params
                    x += dx
                    y += dy
                    absolute_segments.append(['A', rx, ry, angle, large_arc, sweep, x, y])
                elif command == 'z':
                    x = x0
                    y = y0
                    absolute_segments.append(['Z'])

            # Record the start of a subpath
            if command.upper() == 'M':
                x0, y0 = x, y

        # Convert segments back to path data
        path.set('d', self._format_path_segments(absolute_segments))

    def _parse_path(self, d):
        """Parse SVG path data into segments"""
        # This is a simplified parser; in a real implementation you would need
        # a more robust parser that handles all SVG path commands and parameters
        segments = []
        current_segment = []

        # Split path data into tokens
        tokens = []
        i = 0
        while i < len(d):
            if d[i].isalpha():
                tokens.append(d[i])
                i += 1
            elif d[i].isdigit() or d[i] == '-' or d[i] == '.':
                # Parse number
                num_start = i
                i += 1
                while i < len(d) and (d[i].isdigit() or d[i] == '.'):
                    i += 1
                tokens.append(d[num_start:i])
            else:
                # Skip whitespace and other separators
                i += 1

        # Group tokens into segments
        i = 0
        while i < len(tokens):
            if tokens[i].isalpha():
                # Start a new segment
                if current_segment:
                    segments.append(current_segment)
                current_segment = [tokens[i]]
            else:
                # Add number to current segment
                current_segment.append(float(tokens[i]))
            i += 1

        # Add final segment
        if current_segment:
            segments.append(current_segment)

        return segments

    def _format_path_segments(self, segments):
        """Format segments back into SVG path data"""
        path_data = []

        for segment in segments:
            command = segment[0]
            params = segment[1:]

            if command == 'Z':
                path_data.append('Z')
            else:
                # Format parameters
                param_str = ' '.join(str(p) for p in params)
                path_data.append(f"{command} {param_str}")

        return ' '.join(path_data)

    def transform_parse(self, transform_string):
        """Parse SVG transform string into a matrix"""
        if not transform_string:
            return Matrix()

        matrix = Matrix()

        # Regular expressions for parsing transform commands
        import re

        # Extract transform commands and parameters
        transform_regex = r'(matrix|translate|scale|rotate|skewX|skewY)\s*\(\s*([-\d\.\s,]+)\s*\)'
        transforms = re.findall(transform_regex, transform_string)

        for cmd, params_str in transforms:
            # Parse parameters
            params = [float(p) for p in re.split(r'[\s,]+', params_str.strip()) if p]

            if cmd == 'matrix' and len(params) == 6:
                matrix.matrix(params)
            elif cmd == 'translate':
                if len(params) == 1:
                    matrix.translate(params[0], 0)
                elif len(params) == 2:
                    matrix.translate(params[0], params[1])
            elif cmd == 'scale':
                if len(params) == 1:
                    matrix.scale(params[0], params[0])
                elif len(params) == 2:
                    matrix.scale(params[0], params[1])
            elif cmd == 'rotate':
                if len(params) == 1:
                    matrix.rotate(params[0], 0, 0)
                elif len(params) == 3:
                    matrix.rotate(params[0], params[1], params[2])
            elif cmd == 'skewX' and len(params) == 1:
                matrix.skewX(params[0])
            elif cmd == 'skewY' and len(params) == 1:
                matrix.skewY(params[0])

        return matrix

    def apply_transform(self, element, global_transform=''):
        """Recursively apply transform property to the given element"""
        if element is None:
            return

        transform_string = element.get('transform', '')
        transform_string = global_transform + transform_string

        transform = None
        if transform_string and len(transform_string) > 0:
            transform = self.transform_parse(transform_string)

        if not transform:
            transform = Matrix()

        tarray = transform.toArray()

        # Decompose affine matrix to rotate, scale components
        rotate = math.atan2(tarray[1], tarray[3]) * 180 / math.pi
        scale = math.sqrt(tarray[0] * tarray[0] + tarray[2] * tarray[2])

        tag = element.tag.split('}')[-1]  # Get tag without namespace

        if tag in ['g', 'svg', 'defs', 'clipPath']:
            element.attrib.pop('transform', None)

            for child in list(element):
                self.apply_transform(child, transform_string)

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

                # Create a path that represents the ellipse
                d = f"M {cx - rx},{cy} A {rx},{ry} 0 1 0 {cx + rx},{cy} A {rx},{ry} 0 1 0 {cx - rx},{cy} Z"
                path.set('d', d)

                if 'transform' in element.attrib:
                    path.set('transform', element.get('transform'))

                # Replace the element with the path
                parent = element.getparent()
                if parent is not None:
                    index = parent.index(element)
                    parent.insert(index, path)
                    parent.remove(element)

                element = path
                tag = 'path'

            if tag == 'path':
                # Convert path to absolute coordinates
                self.path_to_absolute(element)

                # Get the path data
                d = element.get('d', '')
                if not d:
                    return

                # Parse the path data
                segments = self._parse_path(d)

                # Transform each segment
                transformed_path = []

                prevx, prevy = 0, 0

                for segment in segments:
                    command = segment[0]
                    params = segment[1:]

                    if command == 'H':
                        # Convert H command to L
                        x = params[0]
                        transformed_path.append(['L', x, prevy])
                        prevx, prevy = x, prevy
                    elif command == 'V':
                        # Convert V command to L
                        y = params[0]
                        transformed_path.append(['L', prevx, y])
                        prevx, prevy = prevx, y
                    elif command == 'A':
                        # Transform arc command
                        rx, ry, angle, large_arc, sweep, x, y = params
                        # Apply scale to radius
                        rx *= scale
                        ry *= scale
                        # Apply rotation to angle
                        angle += rotate
                        # Transform end point
                        tx, ty = transform.calc(x, y)
                        transformed_path.append(['A', rx, ry, angle, large_arc, sweep, tx, ty])
                        prevx, prevy = x, y
                    elif command in 'MLCSQT':
                        # Transform all points in the command
                        new_params = []
                        for i in range(0, len(params), 2):
                            x, y = params[i], params[i + 1]
                            tx, ty = transform.calc(x, y)
                            new_params.extend([tx, ty])

                        transformed_path.append([command] + new_params)
                        prevx, prevy = params[-2], params[-1]
                    elif command == 'Z':
                        transformed_path.append([command])

                # Convert segments back to path data
                element.set('d', self._format_path_segments(transformed_path))
                element.attrib.pop('transform', None)

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
        for child in list(element):
            self.flatten(child)

        # Skip the root svg element
        if element.tag.split('}')[-1] != 'svg':
            parent = element.getparent()
            if parent is not None:
                # Move all children to the parent
                for child in list(element):
                    parent.append(child)

    def filter(self, whitelist, element=None):
        """Remove all elements with tag name not in the whitelist"""
        if not whitelist or len(whitelist) == 0:
            raise ValueError('Invalid whitelist')

        element = element or self.svg_root
        if element is None:
            return

        # Process children first (depth-first)
        for child in list(element):
            self.filter(whitelist, child)

        # Check if this element should be removed
        tag = element.tag.split('}')[-1]
        if len(list(element)) == 0 and tag not in whitelist:
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)

    def split_path(self, path):
        """Split a compound path (paths with multiple M commands) into separate paths"""
        if path is None or path.tag.split('}')[-1] != 'path' or path.getparent() is None:
            return False

        # Get the path data
        d = path.get('d', '')
        if not d:
            return False

        # Parse the path data
        segments = self._parse_path(d)

        # Count M commands
        m_count = sum(1 for segment in segments if segment[0] in 'Mm')

        if m_count <= 1:
            return False  # Only 1 M command, no need to split

        # Split into separate paths
        paths = []
        current_path_segments = []

        for segment in segments:
            command = segment[0]

            if command in 'Mm' and current_path_segments:
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
            if len(self._parse_path(new_path.get('d', ''))) > 1:
                parent.insert(parent.index(path), new_path)
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

            p1 = {'x': x, 'y': y}
            p2 = {'x': x + width, 'y': y}
            p3 = {'x': x + width, 'y': y + height}
            p4 = {'x': x, 'y': y + height}

            poly.extend([p1, p2, p3, p4])

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

            # Make sure path is absolute
            self.path_to_absolute(element)

            # Parse the path data
            segments = self._parse_path(d)

            # Process path segments
            x, y = 0, 0
            x0, y0 = 0, 0  # Starting point of current subpath
            x1, y1 = 0, 0  # Control points
            x2, y2 = 0, 0
            prevx, prevy = 0, 0
            prevx1, prevy1 = 0, 0
            prevx2, prevy2 = 0, 0

            for segment in segments:
                command = segment[0]
                params = segment[1:]

                prevx, prevy = x, y
                prevx1, prevy1 = x1, y1
                prevx2, prevy2 = x2, y2

                if command in 'MLHVCSQTA':
                    if command == 'H':
                        x = params[0]
                    elif command == 'V':
                        y = params[0]
                    elif command in 'MCSQT':
                        if len(params) >= 2:
                            x, y = params[-2:]
                        if command == 'C' and len(params) >= 6:
                            x1, y1 = params[0:2]
                            x2, y2 = params[2:4]
                        elif command == 'S' and len(params) >= 4:
                            x2, y2 = params[0:2]
                            if segments:
                                prev_cmd = segments[-1][0]
                                if prev_cmd in 'CS':
                                    # Reflect previous control point
                                    x1 = prevx + (prevx - prevx2)
                                    y1 = prevy + (prevy - prevy2)
                                else:
                                    x1, y1 = prevx, prevy
                        elif command == 'Q' and len(params) >= 4:
                            x1, y1 = params[0:2]
                        elif command == 'T' and len(params) >= 2:
                            if segments:
                                prev_cmd = segments[-1][0]
                                if prev_cmd in 'QT':
                                    # Reflect previous control point
                                    x1 = prevx + (prevx - prevx1)
                                    y1 = prevy + (prevy - prevy1)
                                else:
                                    x1, y1 = prevx, prevy
                    elif command == 'A' and len(params) >= 7:
                        x, y = params[5:7]

                # Process different commands
                if command in 'MLHVmlhv':
                    # Linear segments - add point directly
                    poly.append({'x': x, 'y': y})

                elif command in 'QqTt':
                    # Quadratic Bezier curves - linearize
                    start = {'x': prevx, 'y': prevy}
                    end = {'x': x, 'y': y}
                    control = {'x': x1, 'y': y1}

                    points = GeometryUtil.QuadraticBezier.linearize(start, end, control, self.conf['tolerance'])
                    # Skip first point as it would already be in poly
                    for i in range(1, len(points)):
                        poly.append(points[i])

                elif command in 'CcSs':
                    # Cubic Bezier curves - linearize
                    start = {'x': prevx, 'y': prevy}
                    end = {'x': x, 'y': y}
                    control1 = {'x': x1, 'y': y1}
                    control2 = {'x': x2, 'y': y2}

                    points = GeometryUtil.CubicBezier.linearize(start, end, control1, control2, self.conf['tolerance'])
                    # Skip first point as it would already be in poly
                    for i in range(1, len(points)):
                        poly.append(points[i])

                elif command in 'Aa':
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
                if command in 'Mm':
                    x0, y0 = x, y

                # Close path
                if command in 'Zz':
                    x, y = x0, y0

        # Do not include last point if coincident with starting point
        while (len(poly) > 0 and len(poly) > 1 and
               GeometryUtil.almostEqual(poly[0]['x'], poly[-1]['x'], self.conf['toleranceSvg']) and
               GeometryUtil.almostEqual(poly[0]['y'], poly[-1]['y'], self.conf['toleranceSvg'])):
            poly.pop()

        return poly


# Public API for the SvgParser
def create_parser():
    parser = SvgParser()
    return {
        'config': parser.config,
        'load': parser.load,
        'get_style': parser.get_style,
        'clean': parser.clean_input,
        'polygonify': parser.polygonify
    }

    elif tag == 'rect':
    # Replace rect with polygon
    polygon = etree.Element('{http://www.w3.org/2000/svg}polygon')

    x = float(element.get('x', 0))
    y = float(element.get('y', 0))
    width = float(element.get('width', 0))
    height = float(element.get('height', 0))

    # Create points for the rectangle
    p1 = (x, y)
    p2 = (x + width, y)
    p3 = (x + width, y + height)
    p4 = (x, y + height)

    # Convert points to string
    points_str = f"{p1[0]},{p1[1]} {p2[0]},{p2[1]} {p3[0]},{p3[1]} {p4[0]},{p4[1]}"
    polygon.set('points', points_str)

    if 'transform' in element.attrib:
        polygon.set('transform', element.get('transform'))

    # Replace the element with the polygon
    parent = element.getparent()
    if parent is not None:
        index = parent.index(element)
        parent.insert(index, polygon)
        parent.remove(element)

    element = polygon
    tag = 'polygon'

elif tag in ['polygon', 'polyline']:
# Transform polygon or polyline
points_str = element.get('points', '')
if not points_str:
    return

# Parse points
points = []
for point_str in points_str.split():
    if ',' in point_str:
        x, y = point_str.split(',')
        points.append((float(x), float(y)))

# Transform points
transformed_points = []
for x, y in points:
    tx, ty = transform.calc(x, y)
    transformed_points.append((tx, ty))

# Convert back to string
transformed_points_str = ' '.join(f"{x},{y}" for x, y in transformed_points)
element.set('points', transformed_points_str)

element.attrib.pop('transform', None)

elif tag == 'circle':
# Transform circle
cx = float(element.get('cx', 0))
cy = float(element.get('cy', 0))
r = float(element.get('r', 0))

# Transform center
new_cx, new_cy = transform.calc(cx, cy)

element.set('cx', str(new_cx))
element.set('cy', str(new_cy))

# Apply scale to radius
element.set('r', str(r * scale))

element.attrib.pop('transform', None)

elif tag == 'line':
# Transform line
x1 = float(element.get('x1', 0))
y1 = float(element.get('y1', 0))
x2 = float(element.get('x2', 0))
y2 = float(element.get('y2', 0))

# Transform start and end points
new_x1, new_y1 = transform.calc(x1, y1)
new_x2, new_y2 = transform.calc(x2, y2)

element.set('x1', str(new_x1))
element.set('y1', str(new_y1))
element.set('x2', str(new_x2))
element.set('y2', str(new_y2))

element.attrib.pop('transform', None)