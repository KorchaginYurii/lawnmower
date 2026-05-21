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

    def build(self, env, decomposition, cell_id, start_pos=None):
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
            self.route_index = self.closest_index(start_pos)

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
        while self.route_index < len(self.route):
            x, y = self.route[self.route_index]

            if env.grid[x, y] in (1, 2):
                return (x, y)

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