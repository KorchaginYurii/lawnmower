import numpy as np


class LawnMissionPlanner:
    """
    Планировщик миссии для газонокосилки.

    Стратегия:
    - идти от базы наружу
    - выбирать ближайшие сектора с нескошенной травой
    - не прыгать сразу в дальнюю часть участка
    """

    def __init__(self, sector_h=10, sector_w=10):
        self.sector_h = sector_h
        self.sector_w = sector_w

        self.current_sector = None
        self.route = []
        self.route_index = 0

    def reset(self):
        self.current_sector = None
        self.route = []
        self.route_index = 0

    def sector_id_of(self, pos):
        x, y = pos
        return (
            x // self.sector_h,
            y // self.sector_w,
        )

    def get_sector_bounds(self, sector_id, shape):
        sx, sy = sector_id
        h, w = shape

        x1 = sx * self.sector_h
        x2 = min(h, x1 + self.sector_h)

        y1 = sy * self.sector_w
        y2 = min(w, y1 + self.sector_w)

        return x1, x2, y1, y2

    def all_sector_ids(self, env):
        h, w = env.grid.shape

        ids = []

        for x in range(0, h, self.sector_h):
            for y in range(0, w, self.sector_w):
                ids.append(
                    (
                        x // self.sector_h,
                        y // self.sector_w,
                    )
                )

        return ids

    def sector_uncut(self, env, sector_id):
        x1, x2, y1, y2 = self.get_sector_bounds(
            sector_id,
            env.grid.shape,
        )

        return int(np.sum(env.grid[x1:x2, y1:y2] == 1))

    def sector_cut(self, env, sector_id):
        x1, x2, y1, y2 = self.get_sector_bounds(
            sector_id,
            env.grid.shape,
        )

        return int(np.sum(env.grid[x1:x2, y1:y2] == 2))

    def sector_mowable(self, env, sector_id):
        x1, x2, y1, y2 = self.get_sector_bounds(
            sector_id,
            env.grid.shape,
        )

        sub = env.grid[x1:x2, y1:y2]

        return int(np.sum((sub == 1) | (sub == 2)))

    def sector_coverage(self, env, sector_id):
        mowable = self.sector_mowable(env, sector_id)

        if mowable == 0:
            return 1.0

        return self.sector_cut(env, sector_id) / mowable

    def sector_center(self, env, sector_id):
        x1, x2, y1, y2 = self.get_sector_bounds(
            sector_id,
            env.grid.shape,
        )

        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2,
        )

    def build_route(self, env):
        self.route = []
        self.route_index = 0

        base_sector = self.sector_id_of(env.start_pos)

        sectors = []

        for sector_id in self.all_sector_ids(env):
            uncut = self.sector_uncut(env, sector_id)

            if uncut <= 0:
                continue

            sx, sy = sector_id
            bx, by = base_sector

            dist_from_base = abs(sx - bx) + abs(sy - by)

            center = self.sector_center(env, sector_id)
            px, py = env.pos
            cx, cy = center

            dist_from_robot = abs(cx - px) + abs(cy - py)

            coverage = self.sector_coverage(env, sector_id)

            # Главное:
            # ближние к базе сектора раньше дальних
            score = (
                uncut * 1.0
                - dist_from_base * 80.0
                - dist_from_robot * 0.5
                - coverage * 20.0
            )

            sectors.append(
                {
                    "id": sector_id,
                    "score": score,
                    "uncut": uncut,
                    "dist_base": dist_from_base,
                    "dist_robot" : dist_from_robot,
                }
            )

        sectors.sort(
            key=lambda s: (
                s["score"],
                -s["uncut"],
                -s["score"],
            )
        )

        self.route = [s["id"] for s in sectors]

    def choose_sector(self, env):
        # 1. Если текущий сектор ещё не закончен — остаёмся в нём
        if (
                self.current_sector is not None
                and self.sector_uncut(env, self.current_sector) > 0
        ):
            return self.current_sector

        # 2. Если сектор закончен — пробуем перейти в соседний,
        # а не прыгать через всю карту
        if self.current_sector is not None:
            neighbors = self.neighbor_sectors(env, self.current_sector)

            if neighbors:
                px, py = env.pos

                def score(s):
                    center = self.sector_center(env, s)
                    cx, cy = center
                    uncut = self.sector_uncut(env, s)
                    coverage = self.sector_coverage(env, s)
                    dist = abs(cx - px) + abs(cy - py)

                    return (
                            -dist * 2.0
                            + uncut * 1.0
                            - coverage * 10.0
                    )

                best = max(neighbors, key=score)
                self.current_sector = best
                return best

        # 3. Если это старт или соседей нет — выбираем ближайший к базе сектор
        self.build_route(env)

        if not self.route:
            self.current_sector = None
            return None

        self.current_sector = self.route[0]
        return self.current_sector

    def nearest_uncut_in_sector(self, env, sector_id):
        x1, x2, y1, y2 = self.get_sector_bounds(
            sector_id,
            env.grid.shape,
        )

        cells = np.argwhere(env.grid[x1:x2, y1:y2] == 1)

        if len(cells) == 0:
            return None

        px, py = env.pos

        global_cells = np.array(
            [
                [x + x1, y + y1]
                for x, y in cells
            ]
        )

        dists = np.abs(global_cells - np.array([px, py])).sum(axis=1)

        cell = global_cells[np.argmin(dists)]

        return tuple(map(int, cell))

    def debug(self, env):
        return {
            "sector": self.current_sector,
            "sector_h": self.sector_h,
            "sector_w": self.sector_w,
            "sector_uncut": (
                0
                if self.current_sector is None
                else self.sector_uncut(env, self.current_sector)
            ),
            "sector_coverage": (
                1.0
                if self.current_sector is None
                else self.sector_coverage(env, self.current_sector)
            ),
            "route": self.route,
        }

    def neighbor_sectors(self, env, sector_id):
        sx, sy = sector_id

        candidates = [
            (sx - 1, sy),
            (sx + 1, sy),
            (sx, sy - 1),
            (sx, sy + 1),
        ]

        valid = set(self.all_sector_ids(env))

        return [
            s for s in candidates
            if s in valid and self.sector_uncut(env, s) > 0
        ]