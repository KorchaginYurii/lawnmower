import numpy as np
import random
from collections import deque
from core.config import ACTIONS
from core.energy import EnergySystem
from core.config import DIRECTIONS, MOVE_COST, TURN_COST, CUT_COST
from core.dynamic_obstacles import DynamicObstacleManager
import copy

class CabbageEnv:
    def __init__(self, height=10, width=10):
        self.done = False
        self.flood_cache = {}
        self.max_steps = 200  # дефолт (любое безопасное значение)
        self.obstacle_ratio = 0.0
        self.cabbage_ratio = 0.0
        self.energy_system = EnergySystem(max_energy=100.0)
        self.heading = 0  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.knife_on = False
        self.dynamic_obstacles = DynamicObstacleManager(count=2, move_prob=0.3)
        self.height = height
        self.width = width

    def reset(
            self,
            obs_min=0.05,
            obs_max=0.30,
            cab_min=0.30,
            cab_max=0.70,
            seed=None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.recharge_count = 0

        total = self.height * self.width

        self.obstacle_ratio = random.uniform(obs_min, obs_max)
        self.cabbage_ratio = random.uniform(cab_min, cab_max)
        obstacle_ratio = self.obstacle_ratio
        cabbage_ratio = self.cabbage_ratio
        num_obstacles = int(total * obstacle_ratio)
        num_cabbages = int(total * cabbage_ratio)

        self.energy_system.reset()
        self.heading = random.randint(0, 3)
        self.knife_on = False
        self.allow_start_access = False

        # ===== пустая карта =====
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        h, w = self.grid.shape

        # ===== старт (сначала!) =====
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        self.pos = (x, y)
        self.start_pos = self.pos

        # ===== препятствия =====
        self.obstacles = set()
        free_cells = [(i, j) for i in range(h) for j in range(w)]
        free_cells.remove(self.pos)


        random.shuffle(free_cells)

        for (i, j) in free_cells[:num_obstacles]:
            self.grid[i][j] = -1
            self.obstacles.add((i, j))

        # ===== flood-fill от старта =====
        reachable = self._flood_fill_from_start()

        reachable = [
            c for c in reachable
            if self.grid[c] == 0 and c != self.start_pos
        ]

        if len(reachable) < num_cabbages:
            return self.reset(obs_min, obs_max, cab_min, cab_max)

        random.shuffle(reachable)
        cabbage_cells = reachable[:num_cabbages]

        for (i, j) in cabbage_cells:
            self.grid[i][j] = 1

        reachable_no_start = self.reachable_without_start()
        cabbage_set = set(map(tuple, np.argwhere(self.grid == 1)))

        if not cabbage_set.issubset(reachable_no_start):
            return self.reset(obs_min, obs_max, cab_min, cab_max)

        # ===== остальное =====
        self.visited = np.zeros_like(self.grid, dtype=np.float32)
        self.visit_count = np.zeros_like(self.grid, dtype=np.float32)
        self.turn_count = np.zeros_like(self.grid, dtype=np.float32)
        self.total_turns = 0
        self.energy_used = 0.0

        self.last_positions = deque(maxlen=10)
        self.flood_cache = {}

        self.flood_map = self.compute_flood_map()
        self.danger_map = self.compute_danger_map()
        self.initial_grid = self.grid.copy()

        # ===== расчет MAX_STEP от количества (плотности) капусты
        num_cabbages = np.sum(self.grid == 1)
        num_cabbages = min(num_cabbages, len(reachable) - 1)
        density = num_cabbages / (h * w)

        base = num_cabbages * 4.0
        complexity = np.mean(self.flood_map)

        self.max_steps = int(base + complexity * 10 + max(h,w))

        sx, sy = self.start_pos
        self.grid[sx][sy] = 0
        self.visit_count[sx][sy] += 1

        self.dynamic_obstacles.reset(self)

    def clone(self):
        new = CabbageEnv(self.height, self.width)
        new.height = self.height
        new.width = self.width

        new.grid = self.grid.copy()
        new.pos = tuple(self.pos)
        new.start_pos = tuple(self.start_pos)

        new.steps = self.steps


        new.visited = self.visited.copy()
        new.obstacles = self.obstacles.copy()

        new.last_positions = deque(self.last_positions, maxlen=10)
        new.flood_cache = {}
        new.recharge_count = self.recharge_count

        new.flood_map = self.flood_map.copy()
        new.danger_map = self.danger_map.copy()

        new.obstacle_ratio = self.obstacle_ratio
        new.cabbage_ratio = self.cabbage_ratio

        new.allow_start_access = self.allow_start_access
        new.initial_grid = self.initial_grid.copy()
        new.max_steps = self.max_steps
        new.done = getattr(self, "done", False)
        new.visit_count = self.visit_count.copy()
        new.turn_count = self.turn_count.copy()

        new.total_turns = self.total_turns
        new.energy_used = self.energy_used
        new.heading = self.heading
        new.knife_on = self.knife_on
        new.allow_start_access = self.allow_start_access

        new.energy_system.energy = self.energy_system.energy
        new.energy_system.max_energy = self.energy_system.max_energy
        new.dynamic_obstacles = copy.deepcopy(self.dynamic_obstacles)

        return new

    def step(self, a):

        if hasattr(self, "done") and self.done:
            return 0.0, True


        self.visited *= 0.99
        self.steps += 1


        dx, dy = ACTIONS[a]
        x, y = self.pos
        is_wait = (dx == 0 and dy == 0)

        h, w = self.grid.shape

        nx = max(0, min(h - 1, x + dx))
        ny = max(0, min(w - 1, y + dy))

        energy_spent = 0.0

        if not is_wait:
            target_heading = self.action_direction(a)
            turn_cost = self.turn_cost_to(target_heading)

            if turn_cost > 0:
                self.turn_count[x][y] += 1
                self.total_turns += 1
                self.energy_system.spend(turn_cost)
                energy_spent += turn_cost

            self.heading = target_heading
        else:
            turn_cost = 0

        # reward базовый
        r = -0.05

        # препятствия
        if (nx, ny) in self.obstacles or (nx, ny) in self.dynamic_obstacles.positions():
            nx, ny = x, y
            r -= 1

        # штраф за проход через старт до сбора всей капусты
        rem = np.sum(self.grid == 1) / (np.sum(self.initial_grid == 1) + 1e-6)
        if (
                (nx, ny) == self.start_pos
                and rem > 0
                and not self.allow_start_access
        ):
            nx, ny = x, y
            r -= 0.5

        # энергия за реальное движение
        if not is_wait and (nx, ny) != (x, y):
            self.energy_system.spend_move()
            energy_spent += MOVE_COST

        # штраф за WAIT
        if is_wait:
            r -= 0.02


        # нож только на клетке с капустой
        self.knife_on = ((nx, ny) != (x, y)) and (self.grid[nx][ny] == 1)

        if self.knife_on and self.grid[nx][ny] == 1:
            self.energy_system.spend_cut()
            energy_spent += CUT_COST

        r -= energy_spent

        # капуста
        if self.grid[nx][ny] == 1:
            r += 10
            self.grid[nx][ny] = 0

        # 🔥 anti-loop штраф
        if (nx, ny) in self.last_positions:
            r -= 0.2

        self.pos = (nx, ny)

        self.dynamic_obstacles.step(self)

        # =========================
        # 🔥 ВОЗВРАТ ДОМОЙ (reward shaping)
        # =========================
        remaining = np.sum(self.grid == 1)

        if remaining == 0:
            sx, sy = self.start_pos
            x, y = self.pos

            dist_to_home = abs(x - sx) + abs(y - sy)

            r -= 0.05 * dist_to_home

        # 🔥 обновляем память
        self.last_positions.append((nx, ny))

        r -= 0.1 * self.last_positions.count((nx, ny))

        all_collected = (np.sum(self.grid == 1) == 0)
        at_start = (self.pos == self.start_pos)

        # ===== зарядка на базе =====
        if at_start and np.sum(self.grid == 1) > 0:
            before = self.energy_system.energy

            self.energy_system.recharge()

            after = self.energy_system.energy

            if after > before:
                self.recharge_count += 1
                r += 5

        # ===== reward за возврат =====
        if all_collected:
            sx, sy = self.start_pos
            x, y = self.pos
            dist = abs(x - sx) + abs(y - sy)

            if at_start:
                r += 100
            else:
                r -= 0.1 * dist

        # ===== done =====
        collect_limit = self.max_steps
        h, w = self.grid.shape
        return_limit = max(h,w) * 2

        done = ((all_collected and at_start) or
                (self.steps >= collect_limit + return_limit)
                )

        if all_collected and not at_start:
            r -= 0.2  # заставляет идти домой

        # ===== финальный бонус =====
        if done and all_collected and at_start:
            r += 50
            r += (self.max_steps - self.steps) * 0.5

        r -= 0.05 * self.visited[nx][ny]

        # штраф за “не сдвинулся”
        if (nx, ny) == (x, y):
            r -= 0.3  # или даже -0.5

        # 🔥 penalty за тупики
        r += self.dead_end_penalty((nx, ny))

        r += self.flood_fill_penalty((nx, ny))

        self.visited[nx][ny] = 1
        self.visit_count[nx][ny] += 1

        self.energy_system.energy = max(0.0, self.energy_system.energy)
        if self.energy_system.energy <= 0:
            done = True
            r -= 100

        self.energy_used += energy_spent

        self.done = done

        return r, done

    def dead_end_penalty(self, pos):
        x, y = pos

        blocked = 0
        h, w = self.grid.shape

        for dx, dy in ACTIONS[:4]:
            nx = max(0, min(h, x + dx))
            ny = max(0, min(w, y + dy))

            if (nx, ny) in self.obstacles:
                blocked += 1

        # 🔥 чем больше стен вокруг — тем хуже
        if blocked >= 3:
            return -1.0  # почти тупик
        elif blocked == 2:
            return -0.3  # узкий проход
        else:
            return 0.0

    def flood_fill_area(self, start):
        if start in self.flood_cache:
            return self.flood_cache[start]

        grid = self.grid
        H, W = grid.shape

        free = (grid != -1)

        visited = np.zeros_like(free, dtype=bool)

        x, y = start
        if not free[x, y]:
            return 0

        frontier = np.zeros_like(free, dtype=bool)
        frontier[x, y] = True
        visited[x, y] = True

        while True:
            up = np.roll(frontier, -1, axis=0)
            down = np.roll(frontier, 1, axis=0)
            left = np.roll(frontier, -1, axis=1)
            right = np.roll(frontier, 1, axis=1)

            # убираем wrap-around
            up[0, :] = False
            down[-1, :] = False
            left[:, 0] = False
            right[:, -1] = False

            new_frontier = (up | down | left | right) & free & (~visited)

            if not new_frontier.any():
                break

            visited |= new_frontier
            frontier = new_frontier

        area = visited.sum()
        self.flood_cache[start] = area

        return area

    def flood_fill_penalty(self, pos):
        area = self.flood_fill_area(pos)
        h, w = self.grid.shape
        norm = area / (h * w)

        # 🔥 маленькая область = плохо
        if norm < 0.2:
            return -1.0
        elif norm < 0.4:
            return -0.3
        else:
            return 0.0

    def compute_flood_map(self):
        flood_map = np.zeros_like(self.grid, dtype=np.float32)

        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if self.grid[x, y] != -1:
                    flood_map[x, y] = self.flood_fill_area((x, y))

        return flood_map / (self.grid.size)

    def compute_danger_map(self):
        grid = self.grid
        H, W = grid.shape

        obstacles = (grid == -1).astype(np.int32)

        up = np.roll(obstacles, -1, axis=0)
        down = np.roll(obstacles, 1, axis=0)
        left = np.roll(obstacles, -1, axis=1)
        right = np.roll(obstacles, 1, axis=1)

        # убираем wrap-around
        up[0, :] = 1
        down[-1, :] = 1
        left[:, 0] = 1
        right[:, -1] = 1

        blocked = up + down + left + right

        danger = np.zeros_like(grid, dtype=np.float32)

        danger[blocked >= 3] = -1.0
        danger[blocked == 2] = -0.3

        return danger

    def load_from_file(self, path):
        with open(path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        size = len(lines)

        self.grid = np.zeros((size, size))

        self.obstacles = set()
        self.flood_cache = {}

        for i, line in enumerate(lines):
            for j, ch in enumerate(line):

                if ch == "#":
                    self.grid[i][j] = -1
                    self.obstacles.add((i, j))

                elif ch == "C" or ch == "c":
                    self.grid[i][j] = 1

                elif ch == "A" or ch == "a":
                    self.pos = (i, j)

        if not hasattr(self, "pos"):
            self.pos = (0, 0)

        self.start_pos = self.pos  # запомнить точку старта
        self.visited = np.zeros_like(self.grid)
        self.steps = 0

        from collections import deque
        self.last_positions = deque(maxlen=10)
        self.flood_map = self.compute_flood_map()
        self.danger_map = self.compute_danger_map()

    def _flood_fill_from_start(self):
        visited = set()
        stack = [self.pos]
        h, w = self.grid.shape

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue

            visited.add((x, y))

            for dx, dy in ACTIONS[:4]:
                nx, ny = x + dx, y + dy

                if 0 <= nx < h and 0 <= ny < w:
                    if self.grid[nx][ny] != -1:
                        stack.append((nx, ny))

        return list(visited)

    def action_direction(self, a):
        dx, dy = ACTIONS[a]

        if dx == 0 and dy == 0:
            return self.heading

        for i, (hx, hy) in enumerate(DIRECTIONS):
            if (dx, dy) == (hx, hy):
                return i

        return self.heading

    def turn_cost_to(self, target_heading):
        diff = abs(target_heading - self.heading)
        diff = min(diff, 4 - diff)
        return diff * TURN_COST

    def auto_knife(self, nx, ny):
        """
        Нож включаем только если следующая клетка содержит капусту.
        """
        return self.grid[nx][ny] == 1

    def reachable_without_start(self):
        visited = set()
        stack = [self.pos]
        h, w = self.grid.shape
        while stack:
            x, y = stack.pop()

            if (x, y) in visited:
                continue

            visited.add((x, y))

            for dx, dy in ACTIONS[:4]:
                nx, ny = x + dx, y + dy

                if not (0 <= nx < h and 0 <= ny < w):
                    continue

                if (nx, ny) in self.obstacles:
                    continue
                # старт запрещён как проходная клетка
                if (nx, ny) == self.start_pos:
                    continue

                stack.append((nx, ny))

        return visited