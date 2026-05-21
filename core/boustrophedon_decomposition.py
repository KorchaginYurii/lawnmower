import numpy as np
from collections import deque


class BoustrophedonCell:
    def __init__(self, cell_id):
        self.cell_id = cell_id
        self.cells = set()
        self.frontiers = set()
        self.neighbors = set()

    def add(self, p):
        self.cells.add(p)

    def bbox(self):
        if not self.cells:
            return None

        xs = [p[0] for p in self.cells]
        ys = [p[1] for p in self.cells]

        return (
            min(xs),
            max(xs),
            min(ys),
            max(ys),
        )

    def center(self):
        if not self.cells:
            return None

        xs = [p[0] for p in self.cells]
        ys = [p[1] for p in self.cells]

        return (
            int(sum(xs) / len(xs)),
            int(sum(ys) / len(ys)),
        )


class BoustrophedonDecomposition:
    """
    Simplified Boustrophedon decomposition.

    Splits free space into connected sweep regions.
    """

    def __init__(self):
        self.cells = {}
        self.cell_map = None

    def build(self, env):
        h, w = env.grid.shape

        self.cell_map = -np.ones((h, w), dtype=np.int32)

        visited = set()

        next_cell_id = 0

        for x in range(h):
            for y in range(w):

                if (x, y) in visited:
                    continue

                if env.grid[x, y] not in (1, 2):
                    continue

                region = self.flood_fill_region(
                    env,
                    (x, y),
                    visited,
                )

                if len(region) < 8:
                    continue

                cell = BoustrophedonCell(next_cell_id)

                for p in region:
                    cell.add(p)
                    self.cell_map[p] = next_cell_id

                self.cells[next_cell_id] = cell

                next_cell_id += 1

        self.build_neighbors(env)

    def flood_fill_region(
        self,
        env,
        start,
        visited,
    ):
        q = deque([start])

        region = set([start])

        visited.add(start)

        while q:
            x, y = q.popleft()

            for dx, dy in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
            ]:
                nx = x + dx
                ny = y + dy

                p = (nx, ny)

                if p in visited:
                    continue

                if nx < 0 or ny < 0:
                    continue

                if nx >= env.grid.shape[0]:
                    continue

                if ny >= env.grid.shape[1]:
                    continue

                if env.grid[nx, ny] not in (1, 2):
                    continue

                # =====================================
                # split on narrow passages
                # =====================================
                free_neighbors = 0

                for ddx, ddy in [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                ]:
                    tx = nx + ddx
                    ty = ny + ddy

                    if (
                        0 <= tx < env.grid.shape[0]
                        and 0 <= ty < env.grid.shape[1]
                    ):
                        if env.grid[tx, ty] in (1, 2):
                            free_neighbors += 1

                if free_neighbors <= 1:
                    continue

                visited.add(p)

                region.add(p)

                q.append(p)

        return region

    def build_neighbors(self, env):
        h, w = env.grid.shape

        for x in range(h):
            for y in range(w):

                cid = self.cell_map[x, y]

                if cid < 0:
                    continue

                for dx, dy in [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                ]:
                    nx = x + dx
                    ny = y + dy

                    if nx < 0 or ny < 0:
                        continue

                    if nx >= h or ny >= w:
                        continue

                    nid = self.cell_map[nx, ny]

                    if nid < 0:
                        continue

                    if nid == cid:
                        continue

                    self.cells[cid].neighbors.add(nid)

    def cell_of(self, pos):
        x, y = pos

        if self.cell_map is None:
            return None

        cid = self.cell_map[x, y]

        if cid < 0:
            return None

        return cid

    def debug_info(self):
        return {
            "coverage_cells": len(self.cells),
        }