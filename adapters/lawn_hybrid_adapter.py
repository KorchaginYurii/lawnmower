import numpy as np

from core.config import ACTIONS, WAIT_ACTION
from env.lawn_env import GRASS, CUT, OBSTACLE, BUFFER


class LawnHybridAdapter:
    """
    Адаптер между старым HybridAgent и новой LawnEnv.

    Идея:
    старый агент думает, что:
        grid == 1  -> cabbage

    А в новой задаче:
        grid == 1  -> uncut grass

    Поэтому для агента нескошенная трава выглядит как капуста.
    """

    def __init__(self, lawn_env):
        self.env = lawn_env

        self.grid = self._build_agent_grid()
        self.initial_grid = self.grid.copy()

        self.pos = self.env.pos
        self.start_pos = self.env.start_pos
        self.heading = self.env.heading

        self.energy_system = LawnEnergySystemAdapter(self.env)

        self.obstacles = self._build_obstacles()

        self.knife_on = True
        self.allow_start_access = False

        self.energy_used = self.env.energy_used
        self.visit_count = self.env.visit_count
        self.total_turns = self.env.total_turns

    def sync_from_env(self):
        self.grid = self._build_agent_grid()
        self.pos = self.env.pos
        self.start_pos = self.env.start_pos
        self.heading = self.env.heading

        self.energy_used = self.env.energy_used
        self.visit_count = self.env.visit_count
        self.total_turns = self.env.total_turns

        self.obstacles = self._build_obstacles()
        self.energy_system.sync()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.sync_from_env()
        return reward, done

    def _build_agent_grid(self):
        """
        Для старого агента:
            1  = цель
            0  = свободно
            -1 = препятствие
        """
        g = np.zeros_like(self.env.grid, dtype=np.int16)

        g[self.env.grid == GRASS] = 1
        g[self.env.grid == CUT] = 0
        g[self.env.grid == OBSTACLE] = -1
        g[self.env.grid == BUFFER] = -1

        return g

    def _build_obstacles(self):
        blocked = np.argwhere(
            (self.env.grid == OBSTACLE) |
            (self.env.grid == BUFFER)
        )

        return set(map(tuple, blocked))

    @property
    def dynamic_obstacles(self):
        """
        Чтобы старый HybridAgent не падал, если ждет dynamic_obstacles.
        """
        return EmptyDynamicObstacles()


class LawnEnergySystemAdapter:
    """
    Совместимость с env.energy_system из старого проекта.
    """

    def __init__(self, lawn_env):
        self.env = lawn_env
        self.sync()

    def sync(self):
        self.energy = self.env.energy
        self.max_energy = self.env.max_energy

    def recharge(self):
        self.env.energy = self.env.max_energy
        self.env.energy_used = 0.0
        self.sync()

    def can_reach(self, cost, reserve=0.0):
        self.sync()
        return self.energy >= cost + reserve


class EmptyDynamicObstacles:
    """
    Заглушка для старой логики dynamic obstacles.
    Принимает любые аргументы, чтобы быть совместимой
    с AStarPlanner / HybridAgent.
    """

    def positions(self, *args, **kwargs):
        return set()

    def predicted_positions(self, *args, **kwargs):
        return {}