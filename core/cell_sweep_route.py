class CellSweepRoute:
    """
    Sweep route внутри boustrophedon-cell.
    """

    def __init__(self):
        self.cell_id = None
        self.route = []
        self.route_index = 0

    def reset(self):
        self.cell_id = None
        self.route = []
        self.route_index = 0

    def build(
            self,
            env,
            decomposition,
            cell_id,
            start_pos=None,
            start_policy="lane_start",
    ):
        self.cell_id = cell_id
        self.route = []
        self.route_index = 0

        cell = decomposition.cells[cell_id]
        cells = sorted(cell.cells)

        by_row = {}

        for x, y in cells:
            if env.grid[x, y] in (1, 2):
                by_row.setdefault(x, []).append(y)

        rows = sorted(by_row.keys())

        reverse = False

        for row in rows:
            ys = sorted(by_row[row])

            if reverse:
                ys.reverse()

            for y in ys:
                self.route.append((row, y))

            reverse = not reverse

        if start_pos is not None:
            if start_policy == "nearest":
                self.route_index = self.closest_index(start_pos)
            else:
                self.route_index = self.closest_lane_start_index(start_pos)

    def closest_index(self, pos):
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
        if self.route_index >= len(self.route):
            return

        if env.pos == self.route[self.route_index]:
            self.route_index += 1

    def current_waypoint(self, env):
        """
        Ищем следующую нескошенную клетку впереди по маршруту,
        но не перескакиваем слишком далеко, чтобы сохранить Boustrophedon.
        """

        if not self.route:
            return None

        lookahead = 20

        best_i = None

        end = min(
            len(self.route),
            self.route_index + lookahead,
        )

        for i in range(self.route_index, end):
            x, y = self.route[i]

            if env.grid[x, y] == 1:
                best_i = i
                break

        if best_i is not None:
            self.route_index = best_i
            return self.route[self.route_index]

        # если рядом по маршруту травы нет — тогда уже ищем дальше
        while self.route_index < len(self.route):
            x, y = self.route[self.route_index]

            if env.grid[x, y] == 1:
                return self.route[self.route_index]

            self.route_index += 1

        return None

    def progress(self):
        if not self.route:
            return 1.0

        return self.route_index / max(1, len(self.route))

    def debug(self):
        return {
            "cell_route_len": len(self.route),
            "cell_route_index": self.route_index,
            "cell_route_progress": self.progress(),
            "cell_route_target": (
                None
                if self.route_index >= len(self.route)
                else self.route[self.route_index]
            ),
        }

    def closest_lane_start_index(self, pos):
        """
        Выбирает начало ближайшей ещё не пройденной полосы,
        а не ближайшую точку внутри полосы.
        Это сохраняет нормальную змейку.
        """
        if not self.route:
            return 0

        px, py = pos

        # индексы начала каждого ряда
        lane_starts = [0]

        for i in range(1, len(self.route)):
            prev_x, _ = self.route[i - 1]
            x, _ = self.route[i]

            if x != prev_x:
                lane_starts.append(i)

        best_i = lane_starts[0]
        best_d = 1e18

        for i in lane_starts:
            x, y = self.route[i]
            d = abs(x - px) + abs(y - py)

            if d < best_d:
                best_d = d
                best_i = i

        return best_i

    def set_index_to_target(self, target):
        if not self.route:
            self.route_index = 0
            return

        if target in self.route:
            self.route_index = self.route.index(target)
            return

        self.route_index = self.closest_index(target)