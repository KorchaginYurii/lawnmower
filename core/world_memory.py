import numpy as np
from core.config import ACTIONS

UNKNOWN = -2
EMPTY = 0
CABBAGE = 1
OBSTACLE = -1
BASE = 2


class WorldMemory:
    def __init__(self):
        self.map = None
        self.visited = None
        self.cleaned = None
        self.seen = None

    def reset(self, grid_shape):
        self.map = np.full(grid_shape, UNKNOWN, dtype=np.int8)
        self.visited = np.zeros(grid_shape, dtype=np.float32)
        self.cleaned = np.zeros(grid_shape, dtype=np.float32)
        self.seen = np.zeros(grid_shape, dtype=np.float32)
        self.dynamic_seen = np.zeros(grid_shape, dtype=np.float32)
        self.dynamic_traffic = np.zeros(grid_shape, dtype=np.float32)

    def observe_full(self, env):
        """
        Пока карта полностью известна.
        Позже заменим на observe_local().
        """
        if self.map is None:
            self.reset(env.grid.shape)

        self.map[:, :] = env.grid
        self.map[env.start_pos] = BASE

        self.seen[:, :] = 1.0
        self.visited[env.pos] += 1

    def observe_local(self, env, radius=3):
        """
        Реалистичный вариант: робот видит только локальное окно.
        """
        if self.map is None:
            self.reset(env.grid.shape)

        x, y = env.pos
        h, w = env.grid.shape

        for i in range(max(0, x - radius), min(h, x + radius + 1)):
            for j in range(max(0, y - radius), min(w, y + radius + 1)):
                self.map[i, j] = env.grid[i, j]
                self.seen[i, j] = 1.0

        self.map[env.start_pos] = BASE
        self.visited[env.pos] += 1

        self.dynamic_seen *= 0.90

        if hasattr(env, "dynamic_obstacles"):
            for x, y in env.dynamic_obstacles.positions():
                self.dynamic_seen[x, y] = 1.0

        # traffic slowly decays
        self.dynamic_traffic *= 0.995

        if hasattr(env, "dynamic_obstacles"):
            for x, y in env.dynamic_obstacles.positions():
                self.dynamic_traffic[x, y] += 1.0

    def mark_cleaned(self, pos):
        x, y = pos
        self.cleaned[x, y] = 1.0

        if self.map[x, y] == CABBAGE:
            self.map[x, y] = EMPTY

    def known_obstacles(self):
        return set(map(tuple, np.argwhere(self.map == OBSTACLE)))

    def known_cabbages(self):
        return [tuple(p) for p in np.argwhere(self.map == CABBAGE)]

    def unexplored_cells(self):
        return [tuple(p) for p in np.argwhere(self.seen == 0)]

    def coverage_rate(self):
        return float(np.mean(self.seen))

    def visited_overlap_rate(self):
        visited_cells = np.sum(self.visited > 0)
        overlap_cells = np.sum(self.visited > 1)

        return float(overlap_cells / max(1, visited_cells))

    def in_bounds(self, x, y):
        h, w = self.map.shape
        return 0 <= x < h and 0 <= y < w

    def is_frontier_cell(self, x, y):
        """
        Frontier = известная свободная клетка,
        рядом с которой есть неизвестная клетка.
        """
        if self.map is None:
            return False

        # клетка должна быть известной
        if self.seen[x, y] == 0:
            return False

        # препятствие не frontier
        if self.map[x, y] == OBSTACLE:
            return False

        # рядом должна быть неизвестная клетка
        for dx, dy in ACTIONS:
            nx, ny = x + dx, y + dy

            if not self.in_bounds(nx, ny):
                continue

            if self.seen[nx, ny] == 0:
                return True

        return False

    def frontier_cells(self):
        if self.map is None:
            return []

        h, w = self.map.shape
        frontiers = []

        for x in range(h):
            for y in range(w):
                if self.is_frontier_cell(x, y):
                    frontiers.append((x, y))

        return frontiers

    def nearest_frontier(self, pos, planner=None, env=None):
        frontiers = self.frontier_cells()

        if not frontiers:
            return None

        x, y = pos

        # простой вариант: Manhattan
        if planner is None or env is None:
            dists = [
                abs(fx - x) + abs(fy - y)
                for fx, fy in frontiers
            ]

            return frontiers[int(np.argmin(dists))]

        # более точный вариант: через A*
        best = None
        best_len = 10 ** 9

        for f in frontiers:
            path = planner.find_path_oriented(env, pos, f)

            if path is None:
                continue

            if len(path) < best_len:
                best_len = len(path)
                best = f

        return best

    def copy(self):
        new = WorldMemory()

        new.map = self.map.copy()
        new.seen = self.seen.copy()

        if hasattr(self, "visit_count"):
            new.visit_count = self.visit_count.copy()

        return new

    def dynamic_risk(self, pos):
        if not hasattr(self, "dynamic_seen"):
            return 0.0

        x, y = pos
        return float(self.dynamic_seen[x, y])

    def dynamic_traffic_risk(self, pos):
        if not hasattr(self, "dynamic_traffic"):
            return 0.0

        x, y = pos

        max_v = np.max(self.dynamic_traffic)

        if max_v <= 0:
            return 0.0

        return float(self.dynamic_traffic[x, y] / max_v)
