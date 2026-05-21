import numpy as np


class BoustrophedonMissionPlanner:
    """
    Планировщик natural coverage cells.
    """

    def __init__(self):
        self.current_cell = None
        self.last_cell = None

    def reset(self):
        self.current_cell = None
        self.last_cell = None

    def cell_uncut(self, env, decomposition, cell_id):
        cell = decomposition.cells[cell_id]

        total = 0

        for x, y in cell.cells:
            if env.grid[x, y] == 1:
                total += 1

        return total

    def cell_cut(self, env, decomposition, cell_id):
        cell = decomposition.cells[cell_id]

        total = 0

        for x, y in cell.cells:
            if env.grid[x, y] == 2:
                total += 1

        return total

    def cell_coverage(self, env, decomposition, cell_id):
        cell = decomposition.cells[cell_id]

        cut = 0
        total = 0

        for x, y in cell.cells:
            if env.grid[x, y] in (1, 2):
                total += 1

                if env.grid[x, y] == 2:
                    cut += 1

        if total == 0:
            return 1.0

        return cut / total

    def choose_cell(
        self,
        env,
        decomposition,
    ):
        # ============================================
        # keep current cell if unfinished
        # ============================================

        if self.current_cell is not None:
            if self.cell_uncut(
                env,
                decomposition,
                self.current_cell,
            ) > 0:
                return self.current_cell

        # ============================================
        # try neighbor cells first
        # ============================================

        if self.last_cell is not None:
            neighbors = list(
                decomposition.cells[
                    self.last_cell
                ].neighbors
            )

            best = self.best_cell_from_list(
                env,
                decomposition,
                neighbors,
            )

            if best is not None:
                self.current_cell = best
                return best

        # ============================================
        # global fallback
        # ============================================

        all_cells = list(decomposition.cells.keys())

        best = self.best_cell_from_list(
            env,
            decomposition,
            all_cells,
        )

        self.current_cell = best

        return best

    def best_cell_from_list(
        self,
        env,
        decomposition,
        cells,
    ):
        if not cells:
            return None

        px, py = env.pos

        best = None
        best_score = -1e18

        for cell_id in cells:

            uncut = self.cell_uncut(
                env,
                decomposition,
                cell_id,
            )

            if uncut <= 0:
                continue

            coverage = self.cell_coverage(
                env,
                decomposition,
                cell_id,
            )

            cx, cy = decomposition.cells[
                cell_id
            ].center()

            dist = abs(cx - px) + abs(cy - py)

            visit_penalty = 0.0

            if hasattr(env.env, "visit_count"):
                visit_penalty = (
                    env.env.visit_count[cx, cy]
                    * 2.0
                )

            score = (
                uncut * 3.0
                - coverage * 20.0
                - dist * 1.5
                - visit_penalty
            )

            if score > best_score:
                best_score = score
                best = cell_id

        return best

    def finish_cell(self, cell_id):
        self.last_cell = cell_id

        if self.current_cell == cell_id:
            self.current_cell = None

    def debug(self):
        return {
            "current_cell": self.current_cell,
            "last_cell": self.last_cell,
        }