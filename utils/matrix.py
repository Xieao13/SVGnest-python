# matrix.py

import math
from typing import List, Tuple, Optional

class Matrix:
    """
    Matrix utility for SVG transformations
    Ported from https://github.com/fontello/svgpath
    """
    
    def __init__(self):
        """初始化变换矩阵"""
        self.queue: List[List[float]] = []  # 待应用的矩阵列表
        self.cache: Optional[List[float]] = None  # 组合矩阵缓存

    def combine(self, m1: List[float], m2: List[float]) -> List[float]:
        """
        组合两个矩阵
        m1, m2 - [a, b, c, d, e, f] 格式的矩阵
        """
        return [
            m1[0] * m2[0] + m1[2] * m2[1],
            m1[1] * m2[0] + m1[3] * m2[1],
            m1[0] * m2[2] + m1[2] * m2[3],
            m1[1] * m2[2] + m1[3] * m2[3],
            m1[0] * m2[4] + m1[2] * m2[5] + m1[4],
            m1[1] * m2[4] + m1[3] * m2[5] + m1[5]
        ]

    def is_identity(self) -> bool:
        """检查是否为单位矩阵"""
        if not self.cache:
            self.cache = self.to_array()

        m = self.cache
        return (m[0] == 1 and m[1] == 0 and 
                m[2] == 0 and m[3] == 1 and 
                m[4] == 0 and m[5] == 0)

    def matrix(self, m: List[float]) -> 'Matrix':
        """添加一个变换矩阵到队列"""
        if (m[0] == 1 and m[1] == 0 and 
            m[2] == 0 and m[3] == 1 and 
            m[4] == 0 and m[5] == 0):
            return self

        self.cache = None
        self.queue.append(m)
        return self

    def translate(self, tx: float, ty: float) -> 'Matrix':
        """添加平移变换"""
        if tx != 0 or ty != 0:
            self.cache = None
            self.queue.append([1, 0, 0, 1, tx, ty])
        return self

    def scale(self, sx: float, sy: float) -> 'Matrix':
        """添加缩放变换"""
        if sx != 1 or sy != 1:
            self.cache = None
            self.queue.append([sx, 0, 0, sy, 0, 0])
        return self

    def rotate(self, angle: float, rx: float = 0, ry: float = 0) -> 'Matrix':
        """
        添加旋转变换
        angle: 旋转角度(度)
        rx, ry: 旋转中心点
        """
        if angle != 0:
            self.translate(rx, ry)

            rad = angle * math.pi / 180
            cos = math.cos(rad)
            sin = math.sin(rad)

            self.queue.append([cos, sin, -sin, cos, 0, 0])
            self.cache = None

            self.translate(-rx, -ry)
        return self

    def skew_x(self, angle: float) -> 'Matrix':
        """添加X轴倾斜变换"""
        if angle != 0:
            self.cache = None
            self.queue.append([1, 0, math.tan(angle * math.pi / 180), 1, 0, 0])
        return self

    def skew_y(self, angle: float) -> 'Matrix':
        """添加Y轴倾斜变换"""
        if angle != 0:
            self.cache = None
            self.queue.append([1, math.tan(angle * math.pi / 180), 0, 1, 0, 0])
        return self

    def to_array(self) -> List[float]:
        """将变换队列展平为单个矩阵"""
        if self.cache:
            return self.cache

        if not self.queue:
            self.cache = [1, 0, 0, 1, 0, 0]
            return self.cache

        self.cache = self.queue[0]

        if len(self.queue) == 1:
            return self.cache

        for i in range(1, len(self.queue)):
            self.cache = self.combine(self.cache, self.queue[i])

        return self.cache

    def calc(self, x: float, y: float, is_relative: bool = False) -> Tuple[float, float]:
        """
        将变换应用到点(x,y)
        is_relative: 如果为True,则跳过平移部分
        """
        # 空队列不改变点
        if not self.queue:
            return (x, y)

        # 如果缓存不存在,计算最终矩阵
        if not self.cache:
            self.cache = self.to_array()

        m = self.cache

        # 应用矩阵变换
        return (
            x * m[0] + y * m[2] + (0 if is_relative else m[4]),
            x * m[1] + y * m[3] + (0 if is_relative else m[5])
        )