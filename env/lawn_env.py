import math
import numpy as np

# =========================
# CELL TYPES
# =========================

OBSTACLE = -1
EMPTY = 0
GRASS = 1
CUT = 2
BUFFER = 3

# =========================
# ACTIONS
# =========================

RIGHT = 0
LEFT = 1
DOWN = 2
UP = 3
WAIT = 4

ACTIONS = [
    (0, 1),     # RIGHT
    (0, -1),    # LEFT
    (1, 0),     # DOWN
    (-1, 0),    # UP
    (0, 0),     # WAIT
]

DIRECTIONS = [
    (-1, 0),    # UP
    (0, 1),     # RIGHT
    (1, 0),     # DOWN
    (0, -1),    # LEFT
]


class LawnEnv:
    """
    Среда для газонокосилки.

    Карта:
        -1 = препятствие
         0 = пустая зона, не требует покоса
         1 = трава, надо косить
         2 = уже скошено
         3 = safety buffer вокруг препятствий
    """

    def __init__(
        self,
        width_m=42.0,
        height_m=45.0,
        cell_size_m=0.25,
        robot_size_m=0.50,
        obstacle_inflation_m=0.35,
        move_cost=0.05,
        turn_cost=0.02,
        cut_cost=0.40,
        max_energy=1000.0,
    ):
        self.width_m = width_m
        self.height_m = height_m
        self.cell_size_m = cell_size_m
        self.robot_size_m = robot_size_m
        self.obstacle_inflation_m = obstacle_inflation_m

        self.h = int(round(height_m / cell_size_m))
        self.w = int(round(width_m / cell_size_m))

        self.move_cost = move_cost
        self.turn_cost = turn_cost
        self.cut_cost = cut_cost

        self.max_energy = max_energy
        self.energy = max_energy
        self.energy_used = 0.0

        self.grid = np.zeros((self.h, self.w), dtype=np.int16)
        self.initial_grass = None

        self.pos = (0, 0)
        self.start_pos = (0, 0)
        self.heading = 1  # RIGHT

        self.steps = 0
        self.total_turns = 0

        self.visit_count = np.zeros((self.h, self.w), dtype=np.int32)
        self.cut_count = np.zeros((self.h, self.w), dtype=np.int32)

    # =========================================================
    # MAP INITIALIZATION
    # =========================================================

    def reset_empty_lawn(self, start_pos=None):
        """
        Создает полностью покосную прямоугольную область без препятствий.
        """
        self.grid[:, :] = GRASS

        if start_pos is None:
            start_pos = (self.h - 1, 0)

        self.start_pos = start_pos
        self.pos = start_pos
        self.heading = 1

        self.energy = self.max_energy
        self.energy_used = 0.0
        self.steps = 0
        self.total_turns = 0

        self.visit_count[:, :] = 0
        self.cut_count[:, :] = 0

        self.initial_grass = self.grid == GRASS

        return self.get_state()

    def load_mask(self, mowable_mask, obstacle_mask=None, start_pos=None):
        """
        Загружает карту из масок.

        mowable_mask:
            True там, где надо косить.

        obstacle_mask:
            True там, где препятствие.
        """
        assert mowable_mask.shape == (self.h, self.w)

        self.grid[:, :] = EMPTY
        self.grid[mowable_mask] = GRASS

        if obstacle_mask is not None:
            assert obstacle_mask.shape == (self.h, self.w)
            self.grid[obstacle_mask] = OBSTACLE

        self.inflate_obstacles()

        if start_pos is None:
            start_pos = self.find_first_free_cell()

        self.start_pos = start_pos
        self.pos = start_pos
        self.heading = 1

        self.energy = self.max_energy
        self.energy_used = 0.0
        self.steps = 0
        self.total_turns = 0

        self.visit_count[:, :] = 0
        self.cut_count[:, :] = 0

        self.initial_grass = self.grid == GRASS

        return self.get_state()

    def load_from_txt(self, path, start_pos=None):
        """
        Формат файла:
            -1 = obstacle
             0 = empty
             1 = grass

        Пример строки:
            1 1 1 0 -1 1
        """
        rows = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                rows.append([int(x) for x in line.split()])

        arr = np.array(rows, dtype=np.int16)

        if arr.shape != (self.h, self.w):
            raise ValueError(
                f"Map shape mismatch: got {arr.shape}, expected {(self.h, self.w)}"
            )

        self.grid = arr.copy()
        self.inflate_obstacles()

        if start_pos is None:
            start_pos = self.find_first_free_cell()

        self.start_pos = start_pos
        self.pos = start_pos
        self.heading = 1

        self.energy = self.max_energy
        self.energy_used = 0.0
        self.steps = 0
        self.total_turns = 0

        self.visit_count = np.zeros((self.h, self.w), dtype=np.int32)
        self.cut_count = np.zeros((self.h, self.w), dtype=np.int32)

        self.initial_grass = self.grid == GRASS

        return self.get_state()

    # =========================================================
    # OBSTACLE INFLATION
    # =========================================================

    def inflate_obstacles(self):
        """
        Расширяет препятствия safety buffer'ом.
        Центр робота не должен заходить в buffer.
        """
        obstacle_cells = np.argwhere(self.grid == OBSTACLE)

        inflation_cells = int(
            math.ceil(
                (self.robot_size_m / 2.0 + self.obstacle_inflation_m)
                / self.cell_size_m
            )
        )

        buffer_cells = []

        for ox, oy in obstacle_cells:
            for dx in range(-inflation_cells, inflation_cells + 1):
                for dy in range(-inflation_cells, inflation_cells + 1):
                    nx = ox + dx
                    ny = oy + dy

                    if not self.in_bounds((nx, ny)):
                        continue

                    dist = math.sqrt(dx * dx + dy * dy)

                    if dist <= inflation_cells:
                        if self.grid[nx, ny] != OBSTACLE:
                            buffer_cells.append((nx, ny))

        for x, y in buffer_cells:
            self.grid[x, y] = BUFFER

    # =========================================================
    # ENV STEP
    # =========================================================

    def step(self, action):
        """
        Возвращает:
            state, reward, done, info
        """
        self.steps += 1

        old_pos = self.pos
        old_heading = self.heading

        reward = 0.0
        done = False

        dx, dy = ACTIONS[action]
        nx = old_pos[0] + dx
        ny = old_pos[1] + dy
        new_pos = (nx, ny)

        moved = False

        # -------------------------
        # WAIT
        # -------------------------
        if action == WAIT:
            reward -= 0.01

        # -------------------------
        # MOVE
        # -------------------------
        else:
            if not self.can_stand(new_pos):
                reward -= 1.0
                new_pos = old_pos
            else:
                moved = True
                self.pos = new_pos

                self.energy -= self.move_cost
                self.energy_used += self.move_cost
                reward -= self.move_cost

                new_heading = self.action_to_heading(action)

                turn_diff = self.turn_distance(old_heading, new_heading)

                if turn_diff > 0:
                    turn_energy = turn_diff * self.turn_cost

                    self.energy -= turn_energy
                    self.energy_used += turn_energy
                    reward -= turn_energy

                    self.total_turns += turn_diff

                self.heading = new_heading

        # -------------------------
        # CUTTING
        # -------------------------
        cut_cells = self.cut_under_robot()

        if cut_cells > 0:
            cut_energy = cut_cells * self.cut_cost

            self.energy -= cut_energy
            self.energy_used += cut_energy

            reward += cut_cells * 1.0
            reward -= cut_energy

        # -------------------------
        # VISIT / OVERLAP
        # -------------------------
        x, y = self.pos
        self.visit_count[x, y] += 1

        if self.visit_count[x, y] > 1:
            reward -= 0.02 * self.visit_count[x, y]

        # -------------------------
        # DONE
        # -------------------------
        if self.remaining_grass() == 0:
            done = True
            reward += 100.0

        if self.energy <= 0:
            done = True
            reward -= 100.0

        info = {
            "pos": self.pos,
            "moved": moved,
            "cut_cells": cut_cells,
            "energy": self.energy,
            "energy_used": self.energy_used,
            "coverage_rate": self.coverage_rate(),
            "overlap_rate": self.overlap_rate(),
            "remaining_grass": self.remaining_grass(),
            "total_turns": self.total_turns,
        }

        return self.get_state(), reward, done, info

    # =========================================================
    # CUTTING MODEL
    # =========================================================

    def cut_under_robot(self):
        """
        Робот 0.5 м при cell_size 0.25 м покрывает примерно 2x2 клетки.
        Косим клетки вокруг центра робота.
        """
        radius_cells = max(
            1,
            int(math.ceil((self.robot_size_m / 2.0) / self.cell_size_m))
        )

        cx, cy = self.pos
        cut_cells = 0

        for dx in range(-radius_cells + 1, radius_cells + 1):
            for dy in range(-radius_cells + 1, radius_cells + 1):
                x = cx + dx
                y = cy + dy

                if not self.in_bounds((x, y)):
                    continue

                if self.grid[x, y] == GRASS:
                    self.grid[x, y] = CUT
                    self.cut_count[x, y] += 1
                    cut_cells += 1

                elif self.grid[x, y] == CUT:
                    self.cut_count[x, y] += 1

        return cut_cells

    # =========================================================
    # HELPERS
    # =========================================================

    def get_state(self):
        """
        Базовое состояние.
        Потом сюда добавим каналы для агента.
        """
        return {
            "grid": self.grid.copy(),
            "pos": self.pos,
            "heading": self.heading,
            "energy": self.energy,
            "coverage_rate": self.coverage_rate(),
        }

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.h and 0 <= y < self.w

    def can_stand(self, pos):
        if not self.in_bounds(pos):
            return False

        x, y = pos

        return self.grid[x, y] not in (OBSTACLE, BUFFER)

    def find_first_free_cell(self):
        for x in range(self.h):
            for y in range(self.w):
                if self.can_stand((x, y)):
                    return (x, y)

        raise RuntimeError("No free cell found")

    def action_to_heading(self, action):
        if action == UP:
            return 0
        if action == RIGHT:
            return 1
        if action == DOWN:
            return 2
        if action == LEFT:
            return 3

        return self.heading

    @staticmethod
    def turn_distance(h1, h2):
        diff = abs(h1 - h2)
        return min(diff, 4 - diff)

    def remaining_grass(self):
        return int(np.sum(self.grid == GRASS))

    def total_grass(self):
        if self.initial_grass is None:
            return int(np.sum(self.grid == GRASS))

        return int(np.sum(self.initial_grass))

    def cut_grass(self):
        return self.total_grass() - self.remaining_grass()

    def coverage_rate(self):
        total = self.total_grass()

        if total == 0:
            return 1.0

        return self.cut_grass() / total

    def overlap_rate(self):
        visited = np.sum(self.visit_count > 0)

        if visited == 0:
            return 0.0

        overlap = np.sum(self.visit_count > 1)

        return float(overlap / visited)

    def obstacle_cells(self):
        return set(map(tuple, np.argwhere(self.grid == OBSTACLE)))

    def blocked_cells(self):
        obs = np.argwhere(
            (self.grid == OBSTACLE) | (self.grid == BUFFER)
        )

        return set(map(tuple, obs))

    def grass_cells(self):
        return set(map(tuple, np.argwhere(self.grid == GRASS)))

    def cut_cells(self):
        return set(map(tuple, np.argwhere(self.grid == CUT)))