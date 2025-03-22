import math

# 全局变量
TOL = 10 ** -9  # 浮点数比较的容差

# 辅助函数
def _almostEqual(a, b, tolerance=None):
    if tolerance is None:
        tolerance = TOL
    return abs(a - b) < tolerance

def _withinDistance(p1, p2, distance):
    dx = p1['x'] - p2['x']
    dy = p1['y'] - p2['y']
    return (dx * dx + dy * dy) < distance * distance

def _degreesToRadians(angle):
    return angle * (math.pi / 180)

def _radiansToDegrees(angle):
    return angle * (180 / math.pi)

def _normalizeVector(v):
    if _almostEqual(v['x'] * v['x'] + v['y'] * v['y'], 1):
        return v  # 已经是单位向量
    length = math.sqrt(v['x'] * v['x'] + v['y'] * v['y'])
    inverse = 1 / length
    return {'x': v['x'] * inverse, 'y': v['y'] * inverse}

def _onSegment(A, B, p):
    # 检查点 p 是否在 AB 线段上
    if _almostEqual(A['x'], B['x']) and _almostEqual(p['x'], A['x']):
        if not _almostEqual(p['y'], B['y']) and not _almostEqual(p['y'], A['y']) and p['y'] < max(B['y'], A['y']) and p['y'] > min(B['y'], A['y']):
            return True
        else:
            return False

    if _almostEqual(A['y'], B['y']) and _almostEqual(p['y'], A['y']):
        if not _almostEqual(p['x'], B['x']) and not _almostEqual(p['x'], A['x']) and p['x'] < max(B['x'], A['x']) and p['x'] > min(B['x'], A['x']):
            return True
        else:
            return False

    # 范围检查
    if (p['x'] < A['x'] and p['x'] < B['x']) or (p['x'] > A['x'] and p['x'] > B['x']) or (p['y'] < A['y'] and p['y'] < B['y']) or (p['y'] > A['y'] and p['y'] > B['y']):
        return False

    # 排除端点
    if (_almostEqual(p['x'], A['x']) and _almostEqual(p['y'], A['y'])) or (_almostEqual(p['x'], B['x']) and _almostEqual(p['y'], B['y'])):
        return False

    cross = (p['y'] - A['y']) * (B['x'] - A['x']) - (p['x'] - A['x']) * (B['y'] - A['y'])
    if abs(cross) > TOL:
        return False

    dot = (p['x'] - A['x']) * (B['x'] - A['x']) + (p['y'] - A['y']) * (B['y'] - A['y'])
    if dot < 0 or _almostEqual(dot, 0):
        return False

    len2 = (B['x'] - A['x']) * (B['x'] - A['x']) + (B['y'] - A['y']) * (B['y'] - A['y'])
    if dot > len2 or _almostEqual(dot, len2):
        return False

    return True

def _lineIntersect(A, B, E, F, infinite=False):
    a1 = B['y'] - A['y']
    b1 = A['x'] - B['x']
    c1 = B['x'] * A['y'] - A['x'] * B['y']
    a2 = F['y'] - E['y']
    b2 = E['x'] - F['x']
    c2 = F['x'] * E['y'] - E['x'] * F['y']

    denom = a1 * b2 - a2 * b1
    if denom == 0:
        return None

    x = (b1 * c2 - b2 * c1) / denom
    y = (a2 * c1 - a1 * c2) / denom

    if not math.isfinite(x) or not math.isfinite(y):
        return None

    if not infinite:
        if abs(A['x'] - B['x']) > TOL and ((A['x'] < B['x']) and (x < A['x'] or x > B['x']) or (A['x'] > B['x']) and (x > A['x'] or x < B['x'])):
            return None
        if abs(A['y'] - B['y']) > TOL and ((A['y'] < B['y']) and (y < A['y'] or y > B['y']) or (A['y'] > B['y']) and (y > A['y'] or y < B['y'])):
            return None
        if abs(E['x'] - F['x']) > TOL and ((E['x'] < F['x']) and (x < E['x'] or x > F['x']) or (E['x'] > F['x']) and (x > E['x'] or x < F['x'])):
            return None
        if abs(E['y'] - F['y']) > TOL and ((E['y'] < F['y']) and (y < E['y'] or y > F['y']) or (E['y'] > F['y']) and (y > E['y'] or y < F['y'])):
            return None

    return {'x': x, 'y': y}

# 主类
class GeometryUtil:
    @staticmethod
    def withinDistance(p1, p2, distance):
        return _withinDistance(p1, p2, distance)

    @staticmethod
    def lineIntersect(A, B, E, F, infinite=False):
        return _lineIntersect(A, B, E, F, infinite)

    @staticmethod
    def almostEqual(a, b, tolerance=None):
        return _almostEqual(a, b, tolerance)

    class QuadraticBezier:
        @staticmethod
        def isFlat(p1, p2, c1, tol):
            tol = 4 * tol * tol
            ux = 2 * c1['x'] - p1['x'] - p2['x']
            ux *= ux
            uy = 2 * c1['y'] - p1['y'] - p2['y']
            uy *= uy
            return (ux + uy) <= tol

        @staticmethod
        def linearize(p1, p2, c1, tol):
            finished = [p1]
            todo = [{'p1': p1, 'p2': p2, 'c1': c1}]

            while len(todo) > 0:
                segment = todo[0]
                if GeometryUtil.QuadraticBezier.isFlat(segment['p1'], segment['p2'], segment['c1'], tol):
                    finished.append({'x': segment['p2']['x'], 'y': segment['p2']['y']})
                    todo.pop(0)
                else:
                    divided = GeometryUtil.QuadraticBezier.subdivide(segment['p1'], segment['p2'], segment['c1'], 0.5)
                    todo[0:1] = divided
            return finished

        @staticmethod
        def subdivide(p1, p2, c1, t):
            mid1 = {'x': p1['x'] + (c1['x'] - p1['x']) * t, 'y': p1['y'] + (c1['y'] - p1['y']) * t}
            mid2 = {'x': c1['x'] + (p2['x'] - c1['x']) * t, 'y': c1['y'] + (p2['y'] - c1['y']) * t}
            mid3 = {'x': mid1['x'] + (mid2['x'] - mid1['x']) * t, 'y': mid1['y'] + (mid2['y'] - mid1['y']) * t}
            seg1 = {'p1': p1, 'p2': mid3, 'c1': mid1}
            seg2 = {'p1': mid3, 'p2': p2, 'c1': mid2}
            return [seg1, seg2]

    class CubicBezier:
        @staticmethod
        def isFlat(p1, p2, c1, c2, tol):
            tol = 16 * tol * tol
            ux = 3 * c1['x'] - 2 * p1['x'] - p2['x']
            ux *= ux
            uy = 3 * c1['y'] - 2 * p1['y'] - p2['y']
            uy *= uy
            vx = 3 * c2['x'] - 2 * p2['x'] - p1['x']
            vx *= vx
            vy = 3 * c2['y'] - 2 * p2['y'] - p1['y']
            vy *= vy
            if ux < vx:
                ux = vx
            if uy < vy:
                uy = vy
            return (ux + uy) <= tol

        @staticmethod
        def linearize(p1, p2, c1, c2, tol):
            finished = [p1]
            todo = [{'p1': p1, 'p2': p2, 'c1': c1, 'c2': c2}]

            while len(todo) > 0:
                segment = todo[0]
                if GeometryUtil.CubicBezier.isFlat(segment['p1'], segment['p2'], segment['c1'], segment['c2'], tol):
                    finished.append({'x': segment['p2']['x'], 'y': segment['p2']['y']})
                    todo.pop(0)
                else:
                    divided = GeometryUtil.CubicBezier.subdivide(segment['p1'], segment['p2'], segment['c1'], segment['c2'], 0.5)
                    todo[0:1] = divided
            return finished

        @staticmethod
        def subdivide(p1, p2, c1, c2, t):
            mid1 = {'x': p1['x'] + (c1['x'] - p1['x']) * t, 'y': p1['y'] + (c1['y'] - p1['y']) * t}
            mid2 = {'x': c2['x'] + (p2['x'] - c2['x']) * t, 'y': c2['y'] + (p2['y'] - c2['y']) * t}
            mid3 = {'x': c1['x'] + (c2['x'] - c1['x']) * t, 'y': c1['y'] + (c2['y'] - c1['y']) * t}
            mida = {'x': mid1['x'] + (mid3['x'] - mid1['x']) * t, 'y': mid1['y'] + (mid3['y'] - mid1['y']) * t}
            midb = {'x': mid3['x'] + (mid2['x'] - mid3['x']) * t, 'y': mid3['y'] + (mid2['y'] - mid3['y']) * t}
            midx = {'x': mida['x'] + (midb['x'] - mida['x']) * t, 'y': mida['y'] + (midb['y'] - mida['y']) * t}
            seg1 = {'p1': p1, 'p2': midx, 'c1': mid1, 'c2': mida}
            seg2 = {'p1': midx, 'p2': p2, 'c1': midb, 'c2': mid2}
            return [seg1, seg2]

    class Arc:
        @staticmethod
        def linearize(p1, p2, rx, ry, angle, largearc, sweep, tol):
            finished = [p2]
            arc = GeometryUtil.Arc.svgToCenter(p1, p2, rx, ry, angle, largearc, sweep)
            todo = [arc]

            while len(todo) > 0:
                arc = todo[0]
                fullarc = GeometryUtil.Arc.centerToSvg(arc['center'], arc['rx'], arc['ry'], arc['theta'], arc['extent'], arc['angle'])
                subarc = GeometryUtil.Arc.centerToSvg(arc['center'], arc['rx'], arc['ry'], arc['theta'], 0.5 * arc['extent'], arc['angle'])
                arcmid = subarc['p2']
                mid = {'x': 0.5 * (fullarc['p1']['x'] + fullarc['p2']['x']), 'y': 0.5 * (fullarc['p1']['y'] + fullarc['p2']['y'])}
                if _withinDistance(mid, arcmid, tol):
                    finished.insert(0, fullarc['p2'])
                    todo.pop(0)
                else:
                    arc1 = {'center': arc['center'], 'rx': arc['rx'], 'ry': arc['ry'], 'theta': arc['theta'], 'extent': 0.5 * arc['extent'], 'angle': arc['angle']}
                    arc2 = {'center': arc['center'], 'rx': arc['rx'], 'ry': arc['ry'], 'theta': arc['theta'] + 0.5 * arc['extent'], 'extent': 0.5 * arc['extent'], 'angle': arc['angle']}
                    todo[0:1] = [arc1, arc2]
            return finished

        @staticmethod
        def centerToSvg(center, rx, ry, theta1, extent, angleDegrees):
            theta2 = theta1 + extent
            theta1 = _degreesToRadians(theta1)
            theta2 = _degreesToRadians(theta2)
            angle = _degreesToRadians(angleDegrees)
            cos = math.cos(angle)
            sin = math.sin(angle)
            t1cos = math.cos(theta1)
            t1sin = math.sin(theta1)
            t2cos = math.cos(theta2)
            t2sin = math.sin(theta2)
            x0 = center['x'] + cos * rx * t1cos + (-sin) * ry * t1sin
            y0 = center['y'] + sin * rx * t1cos + cos * ry * t1sin
            x1 = center['x'] + cos * rx * t2cos + (-sin) * ry * t2sin
            y1 = center['y'] + sin * rx * t2cos + cos * ry * t2sin
            largearc = 1 if extent > 180 else 0
            sweep = 1 if extent > 0 else 0
            return {'p1': {'x': x0, 'y': y0}, 'p2': {'x': x1, 'y': y1}, 'rx': rx, 'ry': ry, 'angle': angle, 'largearc': largearc, 'sweep': sweep}

        @staticmethod
        def svgToCenter(p1, p2, rx, ry, angleDegrees, largearc, sweep):
            mid = {'x': 0.5 * (p1['x'] + p2['x']), 'y': 0.5 * (p1['y'] + p2['y'])}
            diff = {'x': 0.5 * (p2['x'] - p1['x']), 'y': 0.5 * (p2['y'] - p1['y'])}
            angle = _degreesToRadians(angleDegrees % 360)
            cos = math.cos(angle)
            sin = math.sin(angle)
            x1 = cos * diff['x'] + sin * diff['y']
            y1 = -sin * diff['x'] + cos * diff['y']
            rx = abs(rx)
            ry = abs(ry)
            Prx = rx * rx
            Pry = ry * ry
            Px1 = x1 * x1
            Py1 = y1 * y1
            radiiCheck = Px1 / Prx + Py1 / Pry
            radiiSqrt = math.sqrt(radiiCheck)
            if radiiCheck > 1:
                rx = radiiSqrt * rx
                ry = radiiSqrt * ry
                Prx = rx * rx
                Pry = ry * ry
            sign = -1 if largearc != sweep else 1
            sq = ((Prx * Pry) - (Prx * Py1) - (Pry * Px1)) / ((Prx * Py1) + (Pry * Px1))
            sq = 0 if sq < 0 else sq
            coef = sign * math.sqrt(sq)
            cx1 = coef * ((rx * y1) / ry)
            cy1 = coef * -((ry * x1) / rx)
            cx = mid['x'] + (cos * cx1 - sin * cy1)
            cy = mid['y'] + (sin * cx1 + cos * cy1)
            ux = (x1 - cx1) / rx
            uy = (y1 - cy1) / ry
            vx = (-x1 - cx1) / rx
            vy = (-y1 - cy1) / ry
            n = math.sqrt((ux * ux) + (uy * uy))
            p = ux
            sign = -1 if uy < 0 else 1
            theta = sign * math.acos(p / n)
            theta = _radiansToDegrees(theta)
            n = math.sqrt((ux * ux + uy * uy) * (vx * vx + vy * vy))
            p = ux * vx + uy * vy
            sign = -1 if (ux * vy - uy * vx) < 0 else 1
            delta = sign * math.acos(p / n)
            delta = _radiansToDegrees(delta)
            if sweep == 1 and delta > 0:
                delta -= 360
            elif sweep == 0 and delta < 0:
                delta += 360
            delta %= 360
            theta %= 360
            return {'center': {'x': cx, 'y': cy}, 'rx': rx, 'ry': ry, 'theta': theta, 'extent': delta,
                    'angle': angleDegrees}

        @staticmethod
        def getPolygonBounds(polygon):
            if not polygon or len(polygon) < 3:
                return None
            xmin = polygon[0]['x']
            xmax = polygon[0]['x']
            ymin = polygon[0]['y']
            ymax = polygon[0]['y']
            for i in range(1, len(polygon)):
                if polygon[i]['x'] > xmax:
                    xmax = polygon[i]['x']
                elif polygon[i]['x'] < xmin:
                    xmin = polygon[i]['x']
                if polygon[i]['y'] > ymax:
                    ymax = polygon[i]['y']
                elif polygon[i]['y'] < ymin:
                    ymin = polygon[i]['y']
            return {'x': xmin, 'y': ymin, 'width': xmax - xmin, 'height': ymax - ymin}

        @staticmethod
        def pointInPolygon(point, polygon):
            if not polygon or len(polygon) < 3:
                return None
            inside = False
            offsetx = polygon.get('offsetx', 0)
            offsety = polygon.get('offsety', 0)
            for i in range(len(polygon)):
                j = len(polygon) - 1 if i == 0 else i - 1
                xi = polygon[i]['x'] + offsetx
                yi = polygon[i]['y'] + offsety
                xj = polygon[j]['x'] + offsetx
                yj = polygon[j]['y'] + offsety
                if _almostEqual(xi, point['x']) and _almostEqual(yi, point['y']):
                    return None
                if _onSegment({'x': xi, 'y': yi}, {'x': xj, 'y': yj}, point):
                    return None
                if _almostEqual(xi, xj) and _almostEqual(yi, yj):
                    continue
                intersect = ((yi > point['y']) != (yj > point['y'])) and (
                            point['x'] < (xj - xi) * (point['y'] - yi) / (yj - yi) + xi)
                if intersect:
                    inside = not inside
            return inside

        @staticmethod
        def polygonArea(polygon):
            area = 0
            for i in range(len(polygon)):
                j = len(polygon) - 1 if i == 0 else i - 1
                area += (polygon[j]['x'] + polygon[i]['x']) * (polygon[j]['y'] - polygon[i]['y'])
            return 0.5 * area

        @staticmethod
        def intersect(A, B):
            Aoffsetx = A.get('offsetx', 0)
            Aoffsety = A.get('offsety', 0)
            Boffsetx = B.get('offsetx', 0)
            Boffsety = B.get('offsety', 0)
            A = A.copy()
            B = B.copy()
            for i in range(len(A) - 1):
                for j in range(len(B) - 1):
                    a1 = {'x': A[i]['x'] + Aoffsetx, 'y': A[i]['y'] + Aoffsety}
                    a2 = {'x': A[i + 1]['x'] + Aoffsetx, 'y': A[i + 1]['y'] + Aoffsety}
                    b1 = {'x': B[j]['x'] + Boffsetx, 'y': B[j]['y'] + Boffsety}
                    b2 = {'x': B[j + 1]['x'] + Boffsetx, 'y': B[j + 1]['y'] + Boffsety}
                    prevbindex = B.length - 1 if j == 0 else j - 1
                    prevaindex = A.length - 1 if i == 0 else i - 1
                    nextbindex = 0 if j + 1 == B.length - 1 else j + 2
                    nextaindex = 0 if i + 1 == A.length - 1 else i + 2
                    if B[prevbindex] == B[j] or (
                            _almostEqual(B[prevbindex]['x'], B[j]['x']) and _almostEqual(B[prevbindex]['y'],
                                                                                         B[j]['y'])):
                        prevbindex = B.length - 1 if prevbindex == 0 else prevbindex - 1
                    if A[prevaindex] == A[i] or (
                            _almostEqual(A[prevaindex]['x'], A[i]['x']) and _almostEqual(A[prevaindex]['y'],
                                                                                         A[i]['y'])):
                        prevaindex = A.length - 1 if prevaindex == 0 else prevaindex - 1
                    if B[nextbindex] == B[j + 1] or (
                            _almostEqual(B[nextbindex]['x'], B[j + 1]['x']) and _almostEqual(B[nextbindex]['y'],
                                                                                             B[j + 1]['y'])):
                        nextbindex = 0 if nextbindex == B.length - 1 else nextbindex + 1
                    if A[nextaindex] == A[i + 1] or (
                            _almostEqual(A[nextaindex]['x'], A[i + 1]['x']) and _almostEqual(A[nextaindex]['y'],
                                                                                             A[i + 1]['y'])):
                        nextaindex = 0 if nextaindex == A.length - 1 else nextaindex + 1
                    a0 = {'x': A[prevaindex]['x'] + Aoffsetx, 'y': A[prevaindex]['y'] + Aoffsety}
                    b0 = {'x': B[prevbindex]['x'] + Boffsetx, 'y': B[prevbindex]['y'] + Boffsety}
                    a3 = {'x': A[nextaindex]['x'] + Aoffsetx, 'y': A[nextaindex]['y'] + Aoffsety}
                    b3 = {'x': B[nextbindex]['x'] + Boffsetx, 'y': B[nextbindex]['y'] + Boffsety}
                    if _onSegment(a1, a2, b1) or (_almostEqual(a1['x'], b1['x']) and _almostEqual(a1['y'], b1['y'])):
                        b0in = GeometryUtil.pointInPolygon(b0, A)
                        b2in = GeometryUtil.pointInPolygon(b2, A)
                        if (b0in == True and b2in == False) or (b0in == False and b2in == True):
                            return True
                        else:
                            continue
                    if _onSegment(a1, a2, b2) or (_almostEqual(a2['x'], b2['x']) and _almostEqual(a2['y'], b2['y'])):
                        b1in = GeometryUtil.pointInPolygon(b1, A)
                        b3in = GeometryUtil.pointInPolygon(b3, A)
                        if (b1in == True and b3in == False) or (b1in == False and b3in == True):
                            return True
                        else:
                            continue
                    if _onSegment(b1, b2, a1) or (_almostEqual(a1['x'], b2['x']) and _almostEqual(a1['y'], b2['y'])):
                        a0in = GeometryUtil.pointInPolygon(a0, B)
                        a2in = GeometryUtil.pointInPolygon(a2, B)
                        if (a0in == True and a2in == False) or (a0in == False and a2in == True):
                            return True
                        else:
                            continue
                    if _onSegment(b1, b2, a2) or (_almostEqual(a2['x'], b1['x']) and _almostEqual(a2['y'], b1['y'])):
                        a1in = GeometryUtil.pointInPolygon(a1, B)
                        a3in = GeometryUtil.pointInPolygon(a3, B)
                        if (a1in == True and a3in == False) or (a1in == False and a3in == True):
                            return True
                        else:
                            continue
                    p = _lineIntersect(b1, b2, a1, a2)
                    if p is not None:
                        return True
            return False

        @staticmethod
        def polygonEdge(polygon, normal):
            if not polygon or len(polygon) < 3:
                return None
            normal = _normalizeVector(normal)
            direction = {'x': -normal['y'], 'y': normal['x']}
            min = None
            max = None
            dotproduct = []
            for i in range(len(polygon)):
                dot = polygon[i]['x'] * direction['x'] + polygon[i]['y'] * direction['y']
                dotproduct.append(dot)
                if min is None or dot < min:
                    min = dot
                if max is None or dot > max:
                    max = dot
            indexmin = 0
            indexmax = 0
            normalmin = None
            normalmax = None
            for i in range(len(polygon)):
                if _almostEqual(dotproduct[i], min):
                    dot = polygon[i]['x'] * normal['x'] + polygon[i]['y'] * normal['y']
                    if normalmin is None or dot > normalmin:
                        normalmin = dot
                        indexmin = i
                elif _almostEqual(dotproduct[i], max):
                    dot = polygon[i]['x'] * normal['x'] + polygon[i]['y'] * normal['y']
                    if normalmax is None or dot > normalmax:
                        normalmax = dot
                        indexmax = i
            indexleft = indexmin - 1
            indexright = indexmin + 1
            if indexleft < 0:
                indexleft = len(polygon) - 1
            if indexright >= len(polygon):
                indexright = 0
            minvertex = polygon[indexmin]
            left = polygon[indexleft]
            right = polygon[indexright]
            leftvector = {'x': left['x'] - minvertex['x'], 'y': left['y'] - minvertex['y']}
            rightvector = {'x': right['x'] - minvertex['x'], 'y': right['y'] - minvertex['y']}
            dotleft = leftvector['x'] * direction['x'] + leftvector['y'] * direction['y']
            dotright = rightvector['x'] * direction['x'] + rightvector['y'] * direction['y']
            scandirection = -1
            if _almostEqual(dotleft, 0):
                scandirection = 1
            elif _almostEqual(dotright, 0):
                scandirection = -1
            else:
                normaldotleft = leftvector['x'] * normal['x'] + leftvector['y'] * normal['y']
                normaldotright = rightvector['x'] * normal['x'] + rightvector['y'] * normal['y']
                if normaldotleft > normaldotright:
                    scandirection = -1
                else:
                    scandirection = 1
            edge = []
            count = 0
            i = indexmin
            while count < len(polygon):
                if i >= len(polygon):
                    i = 0
                elif i < 0:
                    i = len(polygon) - 1
                edge.append(polygon[i])
                if i == indexmax:
                    break
                i += scandirection
                count += 1
            return edge

        @staticmethod
        def pointLineDistance(p, s1, s2, normal, s1inclusive=False, s2inclusive=False):
            normal = _normalizeVector(normal)
            dir = {'x': normal['y'], 'y': -normal['x']}
            pdot = p['x'] * dir['x'] + p['y'] * dir['y']
            s1dot = s1['x'] * dir['x'] + s1['y'] * dir['y']
            s2dot = s2['x'] * dir['x'] + s2['y'] * dir['y']
            pdotnorm = p['x'] * normal['x'] + p['y'] * normal['y']
            s1dotnorm = s1['x'] * normal['x'] + s1['y'] * normal['y']
            s2dotnorm = s2['x'] * normal['x'] + s2['y'] * normal['y']
            if _almostEqual(pdot, s1dot) and _almostEqual(pdot, s2dot):
                if _almostEqual(pdotnorm, s1dotnorm):
                    return None
                if _almostEqual(pdotnorm, s2dotnorm):
                    return None
                if pdotnorm > s1dotnorm and pdotnorm > s2dotnorm:
                    return min(pdotnorm - s1dotnorm, pdotnorm - s2dotnorm)
                if pdotnorm < s1dotnorm and pdotnorm < s2dotnorm:
                    return -min(s1dotnorm - pdotnorm, s2dotnorm - pdotnorm)
                diff1 = pdotnorm - s1dotnorm
                diff2 = pdotnorm - s2dotnorm
                if diff1 > 0:
                    return diff1
                else:
                    return diff2
            elif _almostEqual(pdot, s1dot):
                if s1inclusive:
                    return pdotnorm - s1dotnorm
                else:
                    return None
            elif _almostEqual(pdot, s2dot):
                if s2inclusive:
                    return pdotnorm - s2dotnorm
                else:
                    return None
            elif (pdot < s1dot and pdot < s2dot) or (pdot > s1dot and pdot > s2dot):
                return None
            return pdotnorm - s1dotnorm + (s1dotnorm - s2dotnorm) * (s1dot - pdot) / (s1dot - s2dot)

        @staticmethod
        def pointDistance(p, s1, s2, normal, infinite=False):
            normal = _normalizeVector(normal)
            dir = {'x': normal['y'], 'y': -normal['x']}
            pdot = p['x'] * dir['x'] + p['y'] * dir['y']
            s1dot = s1['x'] * dir['x'] + s1['y'] * dir['y']
            s2dot = s2['x'] * dir['x'] + s2['y'] * dir['y']
            pdotnorm = p['x'] * normal['x'] + p['y'] * normal['y']
            s1dotnorm = s1['x'] * normal['x'] + s1['y'] * normal['y']
            s2dotnorm = s2['x'] * normal['x'] + s2['y'] * normal['y']
            if not infinite:
                if ((pdot < s1dot or _almostEqual(pdot, s1dot)) and (pdot < s2dot or _almostEqual(pdot, s2dot))) or (
                        (pdot > s1dot or _almostEqual(pdot, s1dot)) and (pdot > s2dot or _almostEqual(pdot, s2dot))):
                    return None
                if (_almostEqual(pdot, s1dot) and _almostEqual(pdot, s2dot)) and (
                        pdotnorm > s1dotnorm and pdotnorm > s2dotnorm):
                    return min(pdotnorm - s1dotnorm, pdotnorm - s2dotnorm)
                if (_almostEqual(pdot, s1dot) and _almostEqual(pdot, s2dot)) and (
                        pdotnorm < s1dotnorm and pdotnorm < s2dotnorm):
                    return -min(s1dotnorm - pdotnorm, s2dotnorm - pdotnorm)
            return -(pdotnorm - s1dotnorm + (s1dotnorm - s2dotnorm) * (s1dot - pdot) / (s1dot - s2dot))

        @staticmethod
        def segmentDistance(A, B, E, F, direction):
            normal = {'x': direction['y'], 'y': -direction['x']}
            reverse = {'x': -direction['x'], 'y': -direction['y']}
            dotA = A['x'] * normal['x'] + A['y'] * normal['y']
            dotB = B['x'] * normal['x'] + B['y'] * normal['y']
            dotE = E['x'] * normal['x'] + E['y'] * normal['y']
            dotF = F['x'] * normal['x'] + F['y'] * normal['y']
            crossA = A['x'] * direction['x'] + A['y'] * direction['y']
            crossB = B['x'] * direction['x'] + B['y'] * direction['y']
            crossE = E['x'] * direction['x'] + E['y'] * direction['y']
            crossF = F['x'] * direction['x'] + F['y'] * direction['y']
            crossABmin = min(crossA, crossB)
            crossABmax = max(crossA, crossB)
            crossEFmax = max(crossE, crossF)
            crossEFmin = min(crossE, crossF)
            ABmin = min(dotA, dotB)
            ABmax = max(dotA, dotB)
            EFmax = max(dotE, dotF)
            EFmin = min(dotE, dotF)
            if _almostEqual(ABmax, EFmin, TOL) or _almostEqual(ABmin, EFmax, TOL):
                return None
            if ABmax < EFmin or ABmin > EFmax:
                return None
            overlap = 1 if (ABmax > EFmax and ABmin < EFmin) or (EFmax > ABmax and EFmin < ABmin) else (min(ABmax,
                                                                                                            EFmax) - max(
                ABmin, EFmin)) / (max(ABmax, EFmax) - min(ABmin, EFmin))
            crossABE = (E['y'] - A['y']) * (B['x'] - A['x']) - (E['x'] - A['x']) * (B['y'] - A['y'])
            crossABF = (F['y'] - A['y']) * (B['x'] - A['x']) - (F['x'] - A['x']) * (B['y'] - A['y'])
            if _almostEqual(crossABE, 0) and _almostEqual(crossABF, 0):
                ABnorm = {'x': B['y'] - A['y'], 'y': A['x'] - B['x']}
                EFnorm = {'x': F['y'] - E['y'], 'y': E['x'] - F['x']}
                ABnormlength = math.sqrt(ABnorm['x'] * ABnorm['x'] + ABnorm['y'] * ABnorm['y'])
                ABnorm['x'] /= ABnormlength
                ABnorm['y'] /= ABnormlength
                EFnormlength = math.sqrt(EFnorm['x'] * EFnorm['x'] + EFnorm['y'] * EFnorm['y'])
                EFnorm['x'] /= EFnormlength
                EFnorm['y'] /= EFnormlength
                if abs(ABnorm['y'] * EFnorm['x'] - ABnorm['x'] * EFnorm['y']) < TOL and ABnorm['y'] * EFnorm['y'] + \
                        ABnorm['x'] * EFnorm['x'] < 0:
                    normdot = ABnorm['y'] * direction['y'] + ABnorm['x'] * direction['x']
                    if _almostEqual(normdot, 0, TOL):
                        return None
                    if normdot < 0:
                        return 0
                return None
            distances = []
            if _almostEqual(dotA, dotE):
                distances.append(crossA - crossE)
            elif _almostEqual(dotA, dotF):
                distances.append(crossA - crossF)
            elif dotA > EFmin and dotA < EFmax:
                d = GeometryUtil.pointDistance(A, E, F, reverse)
                if d is not None and _almostEqual(d, 0):
                    dB = GeometryUtil.pointDistance(B, E, F, reverse, True)
                    if dB < 0 or _almostEqual(dB * overlap, 0):
                        d = None
                if d is not None:
                    distances.append(d)
            if _almostEqual(dotB, dotE):
                distances.append(crossB - crossE)
            elif _almostEqual(dotB, dotF):
                distances.append(crossB - crossF)
            elif dotB > EFmin and dotB < EFmax:
                d = GeometryUtil.pointDistance(B, E, F, reverse)
                if d is not None and _almostEqual(d, 0):
                    dA = GeometryUtil.pointDistance(A, E, F, reverse, True)
                    if dA < 0 or _almostEqual(dA * overlap, 0):
                        d = None
                if d is not None:
                    distances.append(d)
            if dotE > ABmin and dotE < ABmax:
                d = GeometryUtil.pointDistance(E, A, B, direction)
                if d is not None and _almostEqual(d, 0):
                    dF = GeometryUtil.pointDistance(F, A, B, direction, True)
                    if dF < 0 or _almostEqual(dF * overlap, 0):
                        d = None
                if d is not None:
                    distances.append(d)
            if dotF > ABmin and dotF < ABmax:
                d = GeometryUtil.pointDistance(F, A, B, direction)
                if d is not None and _almostEqual(d, 0):
                    dE = GeometryUtil.pointDistance(E, A, B, direction, True)
                    if dE < 0 or _almostEqual(dE * overlap, 0):
                        d = None
                if d is not None:
                    distances.append(d)
            if len(distances) == 0:
                return None
            return min(distances)

        @staticmethod
        def polygonSlideDistance(A, B, direction, ignoreNegative=False):
            Aoffsetx = A.get('offsetx', 0)
            Aoffsety = A.get('offsety', 0)
            Boffsetx = B.get('offsetx', 0)
            Boffsety = B.get('offsety', 0)
            A = A.copy()
            B = B.copy()
            if A[0] != A[-1]:
                A.append(A[0])
            if B[0] != B[-1]:
                B.append(B[0])
            edgeA = A
            edgeB = B
            distance = None
            dir = _normalizeVector(direction)
            normal = {'x': dir['y'], 'y': -dir['x']}
            reverse = {'x': -dir['x'], 'y': -dir['y']}
            for i in range(len(edgeB) - 1):
                mind = None
                for j in range(len(edgeA) - 1):
                    A1 = {'x': edgeA[j]['x'] + Aoffsetx, 'y': edgeA[j]['y'] + Aoffsety}
                    A2 = {'x': edgeA[j + 1]['x'] + Aoffsetx, 'y': edgeA[j + 1]['y'] + Aoffsety}
                    B1 = {'x': edgeB[i]['x'] + Boffsetx, 'y': edgeB[i]['y'] + Boffsety}
                    B2 = {'x': edgeB[i + 1]['x'] + Boffsetx, 'y': edgeB[i + 1]['y'] + Boffsety}
                    if (_almostEqual(A1['x'], A2['x']) and _almostEqual(A1['y'], A2['y'])) or (
                            _almostEqual(B1['x'], B2['x']) and _almostEqual(B1['y'], B2['y'])):
                        continue
                    d = GeometryUtil.segmentDistance(A1, A2, B1, B2, dir)
                    if d is not None and (distance is None or d < distance):
                        if not ignoreNegative or d > 0 or _almostEqual(d, 0):
                            distance = d
            return distance

        @staticmethod
        def polygonProjectionDistance(A, B, direction):
            Boffsetx = B.get('offsetx', 0)
            Boffsety = B.get('offsety', 0)
            Aoffsetx = A.get('offsetx', 0)
            Aoffsety = A.get('offsety', 0)
            A = A.copy()
            B = B.copy()
            if A[0] != A[-1]:
                A.append(A[0])
            if B[0] != B[-1]:
                B.append(B[0])
            edgeA = A
            edgeB = B
            distance = None
            for i in range(len(edgeB)):
                minprojection = None
                minp = None
                for j in range(len(edgeA) - 1):
                    p = {'x': edgeB[i]['x'] + Boffsetx, 'y': edgeB[i]['y'] + Boffsety}
                    s1 = {'x': edgeA[j]['x'] + Aoffsetx, 'y': edgeA[j]['y'] + Aoffsety}
                    s2 = {'x': edgeA[j + 1]['x'] + Aoffsetx, 'y': edgeA[j + 1]['y'] + Aoffsety}
                    if abs((s2['y'] - s1['y']) * direction['x'] - (s2['x'] - s1['x']) * direction['y']) < TOL:
                        continue
                    d = GeometryUtil.pointDistance(p, s1, s2, direction)
                    if d is not None and (minprojection is None or d < minprojection):
                        minprojection = d
                        minp = p
                if minprojection is not None and (distance is None or minprojection > distance):
                    distance = minprojection
            return distance

        @staticmethod
        def searchStartPoint(A, B, inside, NFP=None):
            A = A.copy()
            B = B.copy()
            if A[0] != A[-1]:
                A.append(A[0])
            if B[0] != B[-1]:
                B.append(B[0])
            for i in range(len(A) - 1):
                if not A[i].get('marked', False):
                    A[i]['marked'] = True
                    for j in range(len(B)):
                        B['offsetx'] = A[i]['x'] - B[j]['x']
                        B['offsety'] = A[i]['y'] - B[j]['y']
                        Binside = None
                        for k in range(len(B)):
                            inpoly = GeometryUtil.pointInPolygon(
                                {'x': B[k]['x'] + B['offsetx'], 'y': B[k]['y'] + B['offsety']}, A)
                            if inpoly is not None:
                                Binside = inpoly
                                break
                        if Binside is None:
                            return None
                        startPoint = {'x': B['offsetx'], 'y': B['offsety']}
                        if ((Binside and inside) or (not Binside and not inside)) and not GeometryUtil.intersect(A,
                                                                                                                 B) and not inNfp(
                                startPoint, NFP):
                            return startPoint
                        vx = A[i + 1]['x'] - A[i]['x']
                        vy = A[i + 1]['y'] - A[i]['y']
                        d1 = GeometryUtil.polygonProjectionDistance(A, B, {'x': vx, 'y': vy})
                        d2 = GeometryUtil.polygonProjectionDistance(B, A, {'x': -vx, 'y': -vy})
                        d = None
                        if d1 is None and d2 is None:
                            pass
                        elif d1 is None:
                            d = d2
                        elif d2 is None:
                            d = d1
                        else:
                            d = min(d1, d2)
                        if d is not None and not _almostEqual(d, 0) and d > 0:
                            vd2 = vx * vx + vy * vy
                            if d * d < vd2 and not _almostEqual(d * d, vd2):
                                vd = math.sqrt(vx * vx + vy * vy)
                                vx *= d / vd
                                vy *= d / vd
                            B['offsetx'] += vx
                            B['offsety'] += vy
                            for k in range(len(B)):
                                inpoly = GeometryUtil.pointInPolygon(
                                    {'x': B[k]['x'] + B['offsetx'], 'y': B[k]['y'] + B['offsety']}, A)
                                if inpoly is not None:
                                    Binside = inpoly
                                    break
                            startPoint = {'x': B['offsetx'], 'y': B['offsety']}
                            if ((Binside and inside) or (not Binside and not inside)) and not GeometryUtil.intersect(A,
                                                                                                                     B) and not inNfp(
                                    startPoint, NFP):
                                return startPoint

            return None


        @staticmethod
        def isRectangle(poly, tolerance=None):
            bb = GeometryUtil.getPolygonBounds(poly)
            tolerance = tolerance or TOL
            for i in range(len(poly)):
                if not _almostEqual(poly[i]['x'], bb['x']) and not _almostEqual(poly[i]['x'], bb['x'] + bb['width']):
                    return False
                if not _almostEqual(poly[i]['y'], bb['y']) and not _almostEqual(poly[i]['y'], bb['y'] + bb['height']):
                    return False
            return True

        @staticmethod
        def noFitPolygonRectangle(A, B):
            minAx = A[0]['x']
            minAy = A[0]['y']
            maxAx = A[0]['x']
            maxAy = A[0]['y']
            for i in range(1, len(A)):
                if A[i]['x'] < minAx:
                    minAx = A[i]['x']
                if A[i]['y'] < minAy:
                    minAy = A[i]['y']
                if A[i]['x'] > maxAx:
                    maxAx = A[i]['x']
                if A[i]['y'] > maxAy:
                    maxAy = A[i]['y']
            minBx = B[0]['x']
            minBy = B[0]['y']
            maxBx = B[0]['x']
            maxBy = B[0]['y']
            for i in range(1, len(B)):
                if B[i]['x'] < minBx:
                    minBx = B[i]['x']
                if B[i]['y'] < minBy:
                    minBy = B[i]['y']
                if B[i]['x'] > maxBx:
                    maxBx = B[i]['x']
                if B[i]['y'] > maxBy:
                    maxBy = B[i]['y']
            if maxBx - minBx > maxAx - minAx:
                return None
            if maxBy - minBy > maxAy - minAy:
                return None
            return [[
                {'x': minAx - minBx + B[0]['x'], 'y': minAy - minBy + B[0]['y']},
                {'x': maxAx - maxBx + B[0]['x'], 'y': minAy - minBy + B[0]['y']},
                {'x': maxAx - maxBx + B[0]['x'], 'y': maxAy - maxBy + B[0]['y']},
                {'x': minAx - minBx + B[0]['x'], 'y': maxAy - maxBy + B[0]['y']}
            ]]

        @staticmethod
        def noFitPolygon(A, B, inside, searchEdges=False):
            if not A or len(A) < 3 or not B or len(B) < 3:
                return None
            A['offsetx'] = 0
            A['offsety'] = 0
            minA = A[0]['y']
            minAindex = 0
            maxB = B[0]['y']
            maxBindex = 0
            for i in range(1, len(A)):
                A[i]['marked'] = False
                if A[i]['y'] < minA:
                    minA = A[i]['y']
                    minAindex = i
            for i in range(1, len(B)):
                B[i]['marked'] = False
                if B[i]['y'] > maxB:
                    maxB = B[i]['y']
                    maxBindex = i
            if not inside:
                startpoint = {'x': A[minAindex]['x'] - B[maxBindex]['x'], 'y': A[minAindex]['y'] - B[maxBindex]['y']}
            else:
                startpoint = GeometryUtil.searchStartPoint(A, B, True)
            NFPlist = []
            while startpoint is not None:
                B['offsetx'] = startpoint['x']
                B['offsety'] = startpoint['y']
                touching = []
                prevvector = None
                NFP = [{'x': B[0]['x'] + B['offsetx'], 'y': B[0]['y'] + B['offsety']}]
                referencex = B[0]['x'] + B['offsetx']
                referencey = B[0]['y'] + B['offsety']
                startx = referencex
                starty = referencey
                counter = 0
                while counter < 10 * (len(A) + len(B)):
                    touching = []
                    for i in range(len(A)):
                        nexti = 0 if i == len(A) - 1 else i + 1
                        for j in range(len(B)):
                            nextj = 0 if j == len(B) - 1 else j + 1
                            if _almostEqual(A[i]['x'], B[j]['x'] + B['offsetx']) and _almostEqual(A[i]['y'],
                                                                                                  B[j]['y'] + B[
                                                                                                      'offsety']):
                                touching.append({'type': 0, 'A': i, 'B': j})
                            elif _onSegment(A[i], A[nexti],
                                            {'x': B[j]['x'] + B['offsetx'], 'y': B[j]['y'] + B['offsety']}):
                                touching.append({'type': 1, 'A': nexti, 'B': j})
                            elif _onSegment({'x': B[j]['x'] + B['offsetx'], 'y': B[j]['y'] + B['offsety']},
                                            {'x': B[nextj]['x'] + B['offsetx'], 'y': B[nextj]['y'] + B['offsety']},
                                            A[i]):
                                touching.append({'type': 2, 'A': i, 'B': nextj})
                    vectors = []
                    for i in range(len(touching)):
                        vertexA = A[touching[i]['A']]
                        vertexA['marked'] = True
                        prevAindex = len(A) - 1 if touching[i]['A'] == 0 else touching[i]['A'] - 1
                        nextAindex = 0 if touching[i]['A'] == len(A) - 1 else touching[i]['A'] + 1
                        prevA = A[prevAindex]
                        nextA = A[nextAindex]
                        vertexB = B[touching[i]['B']]
                        prevBindex = len(B) - 1 if touching[i]['B'] == 0 else touching[i]['B'] - 1
                        nextBindex = 0 if touching[i]['B'] == len(B) - 1 else touching[i]['B'] + 1
                        prevB = B[prevBindex]
                        nextB = B[nextBindex]
                        if touching[i]['type'] == 0:
                            vA1 = {'x': prevA['x'] - vertexA['x'], 'y': prevA['y'] - vertexA['y'], 'start': vertexA,
                                   'end': prevA}
                            vA2 = {'x': nextA['x'] - vertexA['x'], 'y': nextA['y'] - vertexA['y'], 'start': vertexA,
                                   'end': nextA}
                            vB1 = {'x': vertexB['x'] - prevB['x'], 'y': vertexB['y'] - prevB['y'], 'start': prevB,
                                   'end': vertexB}
                            vB2 = {'x': vertexB['x'] - nextB['x'], 'y': vertexB['y'] - nextB['y'], 'start': nextB,
                                   'end': vertexB}
                            vectors.append(vA1)
                            vectors.append(vA2)
                            vectors.append(vB1)
                            vectors.append(vB2)
                        elif touching[i]['type'] == 1:
                            vectors.append({'x': vertexA['x'] - (vertexB['x'] + B['offsetx']),
                                            'y': vertexA['y'] - (vertexB['y'] + B['offsety']), 'start': prevA,
                                            'end': vertexA})
                            vectors.append({'x': prevA['x'] - (vertexB['x'] + B['offsetx']),
                                            'y': prevA['y'] - (vertexB['y'] + B['offsety']), 'start': vertexA,
                                            'end': prevA})
                        elif touching[i]['type'] == 2:
                            vectors.append({'x': vertexA['x'] - (vertexB['x'] + B['offsetx']),
                                            'y': vertexA['y'] - (vertexB['y'] + B['offsety']), 'start': prevB,
                                            'end': vertexB})
                            vectors.append({'x': vertexA['x'] - (prevB['x'] + B['offsetx']),
                                            'y': vertexA['y'] - (prevB['y'] + B['offsety']), 'start': vertexB,
                                            'end': prevB})
                    translate = None
                    maxd = 0
                    for i in range(len(vectors)):
                        if vectors[i]['x'] == 0 and vectors[i]['y'] == 0:
                            continue
                        if prevvector and vectors[i]['y'] * prevvector['y'] + vectors[i]['x'] * prevvector['x'] < 0:
                            vectorlength = math.sqrt(
                                vectors[i]['x'] * vectors[i]['x'] + vectors[i]['y'] * vectors[i]['y'])
                            unitv = {'x': vectors[i]['x'] / vectorlength, 'y': vectors[i]['y'] / vectorlength}
                            prevlength = math.sqrt(
                                prevvector['x'] * prevvector['x'] + prevvector['y'] * prevvector['y'])
                            prevunit = {'x': prevvector['x'] / prevlength, 'y': prevvector['y'] / prevlength}
                            if abs(unitv['y'] * prevunit['x'] - unitv['x'] * prevunit['y']) < 0.0001:
                                continue
                        d = GeometryUtil.polygonSlideDistance(A, B, vectors[i], True)
                        vecd2 = vectors[i]['x'] * vectors[i]['x'] + vectors[i]['y'] * vectors[i]['y']
                        if d is None or d * d > vecd2:
                            vecd = math.sqrt(vectors[i]['x'] * vectors[i]['x'] + vectors[i]['y'] * vectors[i]['y'])
                            d = vecd
                        if d is not None and d > maxd:
                            maxd = d
                            translate = vectors[i]
                    if translate is None or _almostEqual(maxd, 0):
                        NFP = None
                        break
                    translate['start']['marked'] = True
                    translate['end']['marked'] = True
                    prevvector = translate
                    vlength2 = translate['x'] * translate['x'] + translate['y'] * translate['y']
                    if maxd * maxd < vlength2 and not _almostEqual(maxd * maxd, vlength2):
                        scale = math.sqrt((maxd * maxd) / vlength2)
                        translate['x'] *= scale
                        translate['y'] *= scale
                    referencex += translate['x']
                    referencey += translate['y']
                    if _almostEqual(referencex, startx) and _almostEqual(referencey, starty):
                        break
                    looped = False
                    if len(NFP) > 0:
                        for i in range(len(NFP) - 1):
                            if _almostEqual(referencex, NFP[i]['x']) and _almostEqual(referencey, NFP[i]['y']):
                                looped = True
                    if looped:
                        break
                    NFP.append({'x': referencex, 'y': referencey})
                    B['offsetx'] += translate['x']
                    B['offsety'] += translate['y']
                    counter += 1
                if NFP and len(NFP) > 0:
                    NFPlist.append(NFP)
                if not searchEdges:
                    break
                startpoint = GeometryUtil.searchStartPoint(A, B, inside, NFPlist)
                return NFPlist

            @staticmethod
            def polygonHull(A, B):
                if not A or len(A) < 3 or not B or len(B) < 3:
                    return None
                Aoffsetx = A.get('offsetx', 0)
                Aoffsety = A.get('offsety', 0)
                Boffsetx = B.get('offsetx', 0)
                Boffsety = B.get('offsety', 0)
                miny = A[0]['y']
                startPolygon = A
                startIndex = 0
                for i in range(len(A)):
                    if A[i]['y'] + Aoffsety < miny:
                        miny = A[i]['y'] + Aoffsety
                        startPolygon = A
                        startIndex = i
                for i in range(len(B)):
                    if B[i]['y'] + Boffsety < miny:
                        miny = B[i]['y'] + Boffsety
                        startPolygon = B
                        startIndex = i
                if startPolygon == B:
                    B = A
                    A = startPolygon
                    Aoffsetx = A.get('offsetx', 0)
                    Aoffsety = A.get('offsety', 0)
                    Boffsetx = B.get('offsetx', 0)
                    Boffsety = B.get('offsety', 0)
                A = A.copy()
                B = B.copy()
                C = []
                current = startIndex
                intercept1 = None
                intercept2 = None
                for i in range(len(A) + 1):
                    current = 0 if current == len(A) else current
                    next = 0 if current == len(A) - 1 else current + 1
                    touching = False
                    for j in range(len(B)):
                        nextj = 0 if j == len(B) - 1 else j + 1
                        if _almostEqual(A[current]['x'] + Aoffsetx, B[j]['x'] + Boffsetx) and _almostEqual(
                                A[current]['y'] + Aoffsety, B[j]['y'] + Boffsety):
                            C.append({'x': A[current]['x'] + Aoffsetx, 'y': A[current]['y'] + Aoffsety})
                            intercept1 = j
                            touching = True
                            break
                        elif _onSegment({'x': A[current]['x'] + Aoffsetx, 'y': A[current]['y'] + Aoffsety},
                                        {'x': A[next]['x'] + Aoffsetx, 'y': A[next]['y'] + Aoffsety},
                                        {'x': B[j]['x'] + Boffsetx, 'y': B[j]['y'] + Boffsety}):
                            C.append({'x': A[current]['x'] + Aoffsetx, 'y': A[current]['y'] + Aoffsety})
                            C.append({'x': B[j]['x'] + Boffsetx, 'y': B[j]['y'] + Boffsety})
                            intercept1 = j
                            touching = True
                            break
                        elif _onSegment({'x': B[j]['x'] + Boffsetx, 'y': B[j]['y'] + Boffsety},
                                        {'x': B[nextj]['x'] + Boffsetx, 'y': B[nextj]['y'] + Boffsety},
                                        {'x': A[current]['x'] + Aoffsetx, 'y': A[current]['y'] + Aoffsety}):
                            C.append({'x': A[current]['x'] + Aoffsetx, 'y': A[current]['y'] + Aoffsety})
                            C.append({'x': B[nextj]['x'] + Boffsetx, 'y': B[nextj]['y'] + Boffsety})
                            intercept1 = nextj
                            touching = True
                            break
                    if touching:
                        break
                    C.append({'x': A[current]['x'] + Aoffsetx, 'y': A[current]['y'] + Aoffsety})
                    current += 1
                current = startIndex - 1
                for i in range(len(A) + 1):
                    current = len(A) - 1 if current < 0 else current
                    next = len(A) - 1 if current == 0 else current - 1
                    touching = False
                    for j in range(len(B)):
                        nextj = 0 if j == len(B) - 1 else j + 1
                        if _almostEqual(A[current]['x'] + Aoffsetx, B[j]['x'] + Boffsetx) and _almostEqual(
                                A[current]['y'], B[j]['y'] + Boffsety):
                            C.insert(0, {'x': A[current]['x'] + Aoffsetx, 'y': A[current]['y'] + Aoffsety})
                            intercept2 = j
                            touching = True
                            break
                        elif _onSegment({'x': A[current]['x'] + Aoffsetx, 'y': A[current]['y'] + Aoffsety},
                                        {'x': A[next]['x'] + Aoffsetx, 'y': A[next]['y'] + Aoffsety},
                                        {'x': B[j]['x'] + Boffsetx, 'y': B[j]['y'] + Boffsety}):
                            C.insert(0, {'x': A[current]['x'] + Aoffsetx, 'y': A[current]['y'] + Aoffsety})
                            C.insert(0, {'x': B[j]['x'] + Boffsetx, 'y': B[j]['y'] + Boffsety})
                            intercept2 = j
                            touching = True
                            break
                        elif _onSegment({'x': B[j]['x'] + Boffsetx, 'y': B[j]['y'] + Boffsety},
                                        {'x': B[nextj]['x'] + Boffsetx, 'y': B[nextj]['y'] + Boffsety},
                                        {'x': A[current]['x'] + Aoffsetx, 'y': A[current]['y'] + Aoffsety}):
                            C.insert(0, {'x': A[current]['x'] + Aoffsetx, 'y': A[current]['y'] + Aoffsety})
                            intercept2 = j
                            touching = True
                            break
                    if touching:
                        break
                    C.insert(0, {'x': A[current]['x'] + Aoffsetx, 'y': A[current]['y'] + Aoffsety})
                    current -= 1
                if intercept1 is None or intercept2 is None:
                    return None
                current = intercept1 + 1
                for i in range(len(B)):
                    current = 0 if current == len(B) else current
                    C.append({'x': B[current]['x'] + Boffsetx, 'y': B[current]['y'] + Boffsety})
                    if current == intercept2:
                        break
                    current += 1
                for i in range(len(C)):
                    next = 0 if i == len(C) - 1 else i + 1
                    if _almostEqual(C[i]['x'], C[next]['x']) and _almostEqual(C[i]['y'], C[next]['y']):
                        C.pop(i)
                        i -= 1
                return C

            @staticmethod
            def rotatePolygon(polygon, angle):
                rotated = []
                angle = angle * math.pi / 180
                for i in range(len(polygon)):
                    x = polygon[i]['x']
                    y = polygon[i]['y']
                    x1 = x * math.cos(angle) - y * math.sin(angle)
                    y1 = x * math.sin(angle) + y * math.cos(angle)
                    rotated.append({'x': x1, 'y': y1})
                bounds = GeometryUtil.getPolygonBounds(rotated)
                rotated['x'] = bounds['x']
                rotated['y'] = bounds['y']
                rotated['width'] = bounds['width']
                rotated['height'] = bounds['height']
                return rotated