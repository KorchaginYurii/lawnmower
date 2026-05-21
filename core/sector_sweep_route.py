class SectorSweepRoute:
    """
    Готовый маршрут-змейка внутри сектора.

    Важно:
    - route содержит и GRASS, и CUT клетки
    - CUT клетки НЕ пропускаем автоматически
    - иначе агент начинает возвращаться назад через A*
    - идём по маршруту последовательно: row0 -> row1 -> row2...
    """

    def __init__(self):
        self.sector_id = None
        self.route = []
        self.route_index = 0

    def reset(self):
        self.sector_id = None
        self.route = []
        self.route_index = 0

    def build(self, env, mission, sector_id, start_pos=None):
        self.sector_id = sector_id
        self.route = []
        self.route_index = 0

        x1, x2, y1, y2 = mission.get_sector_bounds(
            sector_id,
            env.grid.shape,
        )

        rows = list(range(x1, x2))

        if start_pos is not None:
            sx, sy = start_pos
            rows.sort(key=lambda r: abs(r - sx))

        reverse = False

        for row in rows:
            cells = []

            if reverse:
                ys = range(y2 - 1, y1 - 1, -1)
            else:
                ys = range(y1, y2)

            for y in ys:
                if env.grid[row, y] in (1, 2):
                    cells.append((row, y))

            if cells:
                self.route.extend(cells)
                reverse = not reverse

        self.route_index = self.find_closest_index(env.pos)

    def find_closest_index(self, pos):
        if not self.route:
            return 0

        px, py = pos

        best_i = 0
        best_d = 10**9

        for i, (x, y) in enumerate(self.route):
            d = abs(x - px) + abs(y - py)

            if d < best_d:
                best_d = d
                best_i = i

        return best_i

    def advance_if_reached(self, env):
        """
        Двигаемся по маршруту последовательно.
        Не пропускаем CUT автоматически.
        """
        if self.route_index >= len(self.route):
            return

        if env.pos == self.route[self.route_index]:
            self.route_index += 1

    def current_waypoint(self, env):
        """
        Следующая физическая точка маршрута.
        Может быть CUT, это нормально.
        """

        while self.route_index < len(self.route):
            x, y = self.route[self.route_index]

            # если клетка стала obstacle/buffer — пропускаем
            if env.grid[x, y] == -1:
                self.route_index += 1
                continue

            # если это empty/non-lawn — пропускаем
            if env.grid[x, y] == 0:
                self.route_index += 1
                continue

            return self.route[self.route_index]

        return None

    def is_finished(self):
        return self.route_index >= len(self.route)

    def progress(self):
        if not self.route:
            return 1.0

        return self.route_index / max(1, len(self.route))

    def debug(self):
        return {
            "sector_route_len": len(self.route),
            "sector_route_index": self.route_index,
            "sector_route_progress": self.progress(),
            "sector_route_target": (
                None
                if self.route_index >= len(self.route)
                else self.route[self.route_index]
            ),
        }