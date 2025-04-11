# genetic_algorithm.py

import math
import random
from typing import List, Dict, Any, Optional
from geometry_util import GeometryUtil
import copy


class Individual:
    """遗传算法中的个体类"""

    def __init__(self, placement: List[Dict] = None, rotation: List[float] = None):
        """初始化个体

        Args:
            placement: 零件放置列表
            rotation: 零件旋转角度列表
        """
        self.placement = placement or []
        self.rotation = rotation or []
        self.fitness = float('inf')  # 默认适应度为无穷大

    def copy(self) -> 'Individual':
        """创建个体的深拷贝"""
        return Individual(
            placement=copy.deepcopy(self.placement),
            rotation=copy.deepcopy(self.rotation)
        )

    def is_valid(self) -> bool:
        """检查个体是否有效"""
        return (isinstance(self.placement, list) and
                isinstance(self.rotation, list) and
                len(self.placement) == len(self.rotation) and
                all(isinstance(r, (int, float)) for r in self.rotation))


class GeneticAlgorithm:
    """遗传算法类，用于优化零件布局"""

    def __init__(self, parts: List[Dict], bin_polygon: List[Dict], config: Dict):
        """
        初始化遗传算法
        parts: 零件列表
        bin_polygon: 容器多边形
        config: 配置参数
        """
        self.parts = parts
        self.bin_polygon = bin_polygon
        self.config = config
        self.population = []
        self.generation_number = 0
        self.bin_bounds = GeometryUtil.get_polygon_bounds(bin_polygon)

        # 初始化种群
        self._init_population()

    def _init_population(self):
        """初始化种群"""
        # 创建初始个体
        adam = Individual()
        adam.placement = self.parts.copy()
        adam.rotation = [0] * len(self.parts)
        self.population = [adam]

        # 生成其他个体
        for _ in range(self.config['populationSize'] - 1):
            individual = Individual()
            individual.placement = self.parts.copy()
            # 随机打乱零件顺序
            random.shuffle(individual.placement)
            # 随机旋转角度
            individual.rotation = [
                random.choice([0, 90, 180, 270])
                for _ in range(len(self.parts))
            ]
            self.population.append(individual)

    def random_angle(self, part: List) -> float:
        """为零件选择随机旋转角度"""
        # 生成可能的角度列表
        angle_list = [i * (360 / max(self.config['rotations'], 1))
                      for i in range(max(self.config['rotations'], 1))]

        # 随机打乱角度列表
        random.shuffle(angle_list)

        # 尝试每个角度，找到合适的
        for angle in angle_list:
            rotated_part = GeometryUtil.rotate_polygon(part, angle)
            bounds = GeometryUtil.get_polygon_bounds(rotated_part)

            # 如果旋转后的零件能放入容器，使用这个角度
            if (bounds['width'] < self.bin_bounds['width'] and
                    bounds['height'] < self.bin_bounds['height']):
                return angle

        return 0

    def mutate(self, individual: Dict) -> Dict:
        """变异操作"""
        clone = {
            'placement': individual['placement'][:],
            'rotation': individual['rotation'][:],
            'fitness': float('inf')
        }

        # 对每个零件
        for i in range(len(clone['placement'])):
            # 有一定概率交换位置
            if random.random() < 0.01 * self.config['mutationRate']:
                j = i + 1
                if j < len(clone['placement']):
                    # 交换位置
                    clone['placement'][i], clone['placement'][j] = \
                        clone['placement'][j], clone['placement'][i]
                    clone['rotation'][i], clone['rotation'][j] = \
                        clone['rotation'][j], clone['rotation'][i]

            # 有一定概率改变旋转角度
            if random.random() < 0.01 * self.config['mutationRate']:
                clone['rotation'][i] = self.random_angle(clone['placement'][i])

        return clone

    def mate(self, male: Dict, female: Dict) -> List[Dict]:
        """交配操作"""
        # 选择交叉点
        cutpoint = round(min(max(random.random(), 0.1), 0.9) *
                         (len(male['placement']) - 1))

        # 创建子代1
        gene1 = male['placement'][:cutpoint]
        rot1 = male['rotation'][:cutpoint]

        # 创建子代2
        gene2 = female['placement'][:cutpoint]
        rot2 = female['rotation'][:cutpoint]

        # 补充剩余基因
        for i in range(len(female['placement'])):
            if not self._contains(gene1, female['placement'][i]['id']):
                gene1.append(female['placement'][i])
                rot1.append(female['rotation'][i])

        for i in range(len(male['placement'])):
            if not self._contains(gene2, male['placement'][i]['id']):
                gene2.append(male['placement'][i])
                rot2.append(male['rotation'][i])

        return [
            {'placement': gene1, 'rotation': rot1, 'fitness': float('inf')},
            {'placement': gene2, 'rotation': rot2, 'fitness': float('inf')}
        ]

    def _contains(self, gene: List, id: int) -> bool:
        """检查基因序列是否包含指定ID的零件"""
        return any(part['id'] == id for part in gene)

    def generation(self) -> List[Individual]:
        """进行一代进化

        Returns:
            List[Individual]: 新一代种群
        """
        # 按适应度排序
        self.population.sort(key=lambda x: x.fitness)

        # 保留精英个体
        new_population = [self.population[0].copy()]

        # 生成新一代
        while len(new_population) < self.config['populationSize']:
            # 选择父代
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            # 交叉
            child = self._crossover(parent1, parent2)

            # 变异
            if random.random() < self.config['mutationRate'] / 100:
                child = self._mutate(child)

            new_population.append(child)

        self.population = new_population
        self.generation_number += 1
        return self.population

    def _tournament_select(self) -> Individual:
        """锦标赛选择

        Returns:
            Individual: 选中的个体
        """
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return min(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """交叉操作

        Args:
            parent1: 父代1
            parent2: 父代2

        Returns:
            Individual: 子代
        """
        child = Individual()

        # 随机选择交叉点
        crossover_point = random.randint(0, len(parent1.placement))

        # 从父代1复制前半部分
        child.placement = parent1.placement[:crossover_point]
        child.rotation = parent1.rotation[:crossover_point]

        # 从父代2复制剩余部分（避免重复）
        remaining_parts = [
            part for part in parent2.placement
            if part not in child.placement
        ]
        remaining_rotations = parent2.rotation[len(child.placement):]

        child.placement.extend(remaining_parts)
        child.rotation.extend(remaining_rotations)

        return child

    def _mutate(self, individual: Individual) -> Individual:
        """变异操作

        Args:
            individual: 待变异个体

        Returns:
            Individual: 变异后的个体
        """
        mutant = individual.copy()

        # 随机选择两个位置交换
        i, j = random.sample(range(len(mutant.placement)), 2)
        mutant.placement[i], mutant.placement[j] = mutant.placement[j], mutant.placement[i]

        # 随机改变一个旋转角度
        k = random.randint(0, len(mutant.rotation) - 1)
        mutant.rotation[k] = random.choice([0, 90, 180, 270])

        return mutant

    def _random_weighted_individual(self, exclude: Optional[Dict] = None) -> Dict:
        """
        从种群中随机选择个体，前面的个体（适应度更好）有更高的选择概率
        exclude: 要排除的个体
        """
        pop = self.population[:]
        if exclude and exclude in pop:
            pop.remove(exclude)

        rand = random.random()

        # 计算权重
        weight = 1 / len(pop)
        lower = 0
        upper = weight

        for i, individual in enumerate(pop):
            if rand > lower and rand < upper:
                return individual
            lower = upper
            upper += 2 * weight * ((len(pop) - i) / len(pop))

        return pop[0]
