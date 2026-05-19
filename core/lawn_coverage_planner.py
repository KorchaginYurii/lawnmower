import numpy as np


class LawnCoveragePlanner:
    """
    Coverage planner для газонокосилки.

    Главная идея:
    - не искать ближайшую траву как отдельную цель
    - строить sweep-маршрут внутри сектора
    - двигаться полосами: туда-обратно
    """

    def __init__(self):
        self.route = []
        self.route_index = 0
        self.cached_signature = None
        self.cached_sector = None
        self.sweep_direction = "horizontal"

    def reset(self):
        self.route = []
        self.route_index = 0
        self.cached_signature = None
        self.cached_sector = None

    # =====================================================
    # PUBLIC API
    # =====================================================

    def get_next_target(
        self,
        env,
        sector_manager=None,
        sector_id=None,
        memory=None,
        prev_pos=None,
    ):
        """
        Главный метод.

        Возвращает следующую клетку травы, которую надо косить.
        """

        signature = self.build_signature(env, sector_id, sector_manager)

        need_rebuild = (
            len(self.route) == 0
            or self.route_index >= len(self.route)
            or signature != self.cached_signature
            or sector_id != self.cached_sector
        )

        if need_rebuild:
            self.build_sweep_route(
                env=env,
                sector_manager=sector_manager,
                sector_id=sector_id,
                memory=memory,
                start_pos=env.pos,
                prev_pos=prev_pos,
            )

        while self.route_index < len(self.route):
            target = self.route[self.route_index]

            if self.is_uncut(env, target):
                return target

            self.route_index += 1

        return self.find_nearest_uncut(env)

    # =====================================================
    # ROUTE BUILDING
    # =====================================================

    def build_sweep_route(
        self,
        env,
        sector_manager=None,
        sector_id=None,
        memory=None,
        start_pos=None,
        prev_pos=None,
    ):
        self.route = []
        self.route_index = 0

        cells = self.collect_uncut_cells(
            env,
            sector_manager=sector_manager,
            sector_id=sector_id,
        )

        if len(cells) == 0:
            self.cached_signature = self.build_signature(
                env,
                sector_id,
                sector_manager,
            )
            self.cached_sector = sector_id
            return

        direction = self.find_best_sweep_direction(cells)
        self.sweep_direction = direction

        if direction == "horizontal":
            route = self.build_horizontal_sweep(cells, start_pos)
        else:
            route = self.build_vertical_sweep(cells, start_pos)

        route = self.remove_unreachable_noise(env, route)

        self.route = route
        self.cached_signature = self.build_signature(
            env,
            sector_id,
            sector_manager,
        )
        self.cached_sector = sector_id

    def build_horizontal_sweep(self, cells, start_pos=None):
        """
        Строит маршрут строками:

        row 1: left -> right
        row 2: right -> left
        row 3: left -> right
        """

        by_row = {}

        for x, y in cells:
            by_row.setdefault(x, []).append(y)

        rows = sorted(by_row.keys())

        if start_pos is not None:
            sx, sy = start_pos
            rows = sorted(rows, key=lambda r: abs(r - sx))

        route = []
        reverse = False

        for row in rows:
            ys = sorted(by_row[row])

            segments = self.split_continuous_segments(ys)

            if reverse:
                segments = list(reversed(segments))

            for seg in segments:
                if reverse:
                    seg = list(reversed(seg))

                for y in seg:
                    route.append((row, y))

            reverse = not reverse

        return route

    def build_vertical_sweep(self, cells, start_pos=None):
        """
        Строит маршрут колонками:

        col 1: top -> bottom
        col 2: bottom -> top
        col 3: top -> bottom
        """

        by_col = {}

        for x, y in cells:
            by_col.setdefault(y, []).append(x)

        cols = sorted(by_col.keys())

        if start_pos is not None:
            sx, sy = start_pos
            cols = sorted(cols, key=lambda c: abs(c - sy))

        route = []
        reverse = False

        for col in cols:
            xs = sorted(by_col[col])

            segments = self.split_continuous_segments(xs)

            if reverse:
                segments = list(reversed(segments))

            for seg in segments:
                if reverse:
                    seg = list(reversed(seg))

                for x in seg:
                    route.append((x, col))

            reverse = not reverse

        return route

    # =====================================================
    # SWEEP DIRECTION
    # =====================================================

    def find_best_sweep_direction(self, cells):
        """
        Выбираем направление с меньшим количеством полос.

        Если участок шире, лучше идти горизонтально.
        Если участок выше, лучше идти вертикально.
        """

        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]

        height = max(xs) - min(xs) + 1
        width = max(ys) - min(ys) + 1

        rows = len(set(xs))
        cols = len(set(ys))

        horizontal_score = rows
        vertical_score = cols

        if width >= height:
            horizontal_score *= 0.9
        else:
            vertical_score *= 0.9

        if horizontal_score <= vertical_score:
            return "horizontal"

        return "vertical"

    # =====================================================
    # CELL COLLECTION
    # =====================================================

    def collect_uncut_cells(
        self,
        env,
        sector_manager=None,
        sector_id=None,
    ):
        """
        Собирает клетки травы в секторе или на всей карте.

        В adapter-env:
            env.grid == 1 означает нескошенную траву.
        """

        if sector_manager is not None and sector_id is not None:
            x1, x2, y1, y2 = sector_manager.get_sector_bounds(
                sector_id,
                env.grid.shape,
            )

            sub = env.grid[x1:x2, y1:y2]
            local = np.argwhere(sub == 1)

            cells = [
                (int(x + x1), int(y + y1))
                for x, y in local
            ]

            return cells

        cells = np.argwhere(env.grid == 1)

        return [
            (int(x), int(y))
            for x, y in cells
        ]

    def is_uncut(self, env, pos):
        x, y = pos

        if x < 0 or y < 0:
            return False

        if x >= env.grid.shape[0] or y >= env.grid.shape[1]:
            return False

        return env.grid[x, y] == 1

    # =====================================================
    # FALLBACKS
    # =====================================================

    def find_nearest_uncut(self, env):
        cells = np.argwhere(env.grid == 1)

        if len(cells) == 0:
            return None

        x, y = env.pos

        dists = np.abs(cells - np.array([x, y])).sum(axis=1)
        nearest = cells[np.argmin(dists)]

        return tuple(map(int, nearest))

    def remove_unreachable_noise(self, env, route):
        """
        Мягкий фильтр:
        убираем точки вне карты и явные препятствия.
        Реальную достижимость всё равно проверит A*.
        """

        clean = []

        h, w = env.grid.shape

        obstacles = getattr(env, "obstacles", set())

        for x, y in route:
            if x < 0 or y < 0 or x >= h or y >= w:
                continue

            if (x, y) in obstacles:
                continue

            clean.append((x, y))

        return clean

    # =====================================================
    # UTILS
    # =====================================================

    def split_continuous_segments(self, values):
        """
        [1,2,3,7,8,10] ->
        [[1,2,3], [7,8], [10]]

        Нужно для препятствий внутри строки/колонки.
        """

        if len(values) == 0:
            return []

        values = sorted(values)

        segments = []
        current = [values[0]]

        for v in values[1:]:
            if v == current[-1] + 1:
                current.append(v)
            else:
                segments.append(current)
                current = [v]

        segments.append(current)

        return segments

    def build_signature(
        self,
        env,
        sector_id=None,
        sector_manager=None,
    ):
        """
        Сигнатура нужна, чтобы понимать:
        изменилась ли карта травы.
        """

        if sector_manager is not None and sector_id is not None:
            x1, x2, y1, y2 = sector_manager.get_sector_bounds(
                sector_id,
                env.grid.shape,
            )

            return int(np.sum(env.grid[x1:x2, y1:y2] == 1))

        return int(np.sum(env.grid == 1))