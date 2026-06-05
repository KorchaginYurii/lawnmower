import math
import random
import numpy as np

from env.lawn_env import GRASS, EMPTY, OBSTACLE


class LawnMapGenerator:
    def __init__(
            self,
            h,
            w,
            min_object_size=4,
            max_object_size=24,
            seed=None,
    ):
        self.h = h
        self.w = w

        safe_max = max(2, min(max_object_size, h // 4, w // 4))

        self.min_object_size = min(min_object_size, safe_max)
        self.max_object_size = safe_max
        self.reserved_zones = []
        self.start_pos = None

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_realistic_lawn(
        self,
        object_count=8,
        border_margin=4,
    ):
        grid = np.full((self.h, self.w), GRASS, dtype=np.int16)

        # внешняя граница как нерабочая зона
        grid[:border_margin, :] = EMPTY
        grid[-border_margin:, :] = EMPTY
        grid[:, :border_margin] = EMPTY
        grid[:, -border_margin:] = EMPTY

        self.start_pos = self.add_home_block(grid)

        for _ in range(object_count):
            shape = random.choice(["circle", "ellipse", "rect", "blob"])

            if shape == "circle":
                self.add_circle(grid)

            elif shape == "ellipse":
                self.add_ellipse(grid)

            elif shape == "rect":
                self.add_rect(grid)

            else:
                self.add_blob(grid)


        return grid

    def add_circle(self, grid, attempts=30):
        for _ in range(attempts):
            r = random.randint(
                self.min_object_size,
                max(self.min_object_size, self.max_object_size // 2),
            )

            r = min(
                r,
                max(2, self.h // 4),
                max(2, self.w // 4),
            )

            # объект не помещается
            if self.h - r - 3 <= r + 2:
                return

            if self.w - r - 3 <= r + 2:
                return

            cx = random.randint(r + 2, self.h - r - 3)
            cy = random.randint(r + 2, self.w - r - 3)

            # bbox конкретного круга
            x1 = cx - r
            x2 = cx + r + 1
            y1 = cy - r
            y2 = cy + r + 1

            if self.intersects_reserved(x1, x2, y1, y2):
                continue

            for x in range(x1, x2):
                for y in range(y1, y2):
                    if 0 <= x < self.h and 0 <= y < self.w:
                        if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                            grid[x, y] = OBSTACLE

            return

    def add_ellipse(self, grid, attempts=30):
        for _ in range(attempts):
            rx = random.randint(
                self.min_object_size,
                max(self.min_object_size, self.max_object_size),
            )

            ry = random.randint(
                self.min_object_size,
                max(self.min_object_size, self.max_object_size),
            )

            rx = min(rx, max(2, self.h // 4))
            ry = min(ry, max(2, self.w // 4))

            if self.h - rx - 3 <= rx + 2:
                return

            if self.w - ry - 3 <= ry + 2:
                return

            cx = random.randint(rx + 2, self.h - rx - 3)
            cy = random.randint(ry + 2, self.w - ry - 3)

            # bbox конкретного эллипса
            x1 = cx - rx
            x2 = cx + rx + 1
            y1 = cy - ry
            y2 = cy + ry + 1

            if self.intersects_reserved(x1, x2, y1, y2):
                continue

            for x in range(x1, x2):
                for y in range(y1, y2):
                    if 0 <= x < self.h and 0 <= y < self.w:
                        v = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
                        if v <= 1.0:
                            grid[x, y] = OBSTACLE
            return

    def add_rect(self, grid, attempts=30):
        for _ in range(attempts):
            hh = random.randint(
                self.min_object_size,
                self.max_object_size,
            )
            ww = random.randint(
                self.min_object_size,
                self.max_object_size,
            )

            if self.h - hh - 3 <= 2:
                continue

            if self.w - ww - 3 <= 2:
                continue

            x1 = random.randint(2, self.h - hh - 3)
            y1 = random.randint(2, self.w - ww - 3)

            x2 = x1 + hh
            y2 = y1 + ww

            if self.intersects_reserved(x1, x2, y1, y2):
                continue

            grid[x1:x2, y1:y2] = OBSTACLE
            return

    def add_blob(self, grid, attempts=30):
        for _ in range(attempts):
            cx = random.randint(6, self.h - 7)
            cy = random.randint(6, self.w - 7)

            points = random.randint(12, 30)

            radius = random.randint(
                self.min_object_size,
                self.max_object_size,
            )

            x1 = cx - radius - 5
            x2 = cx + radius + 6
            y1 = cy - radius - 5
            y2 = cy + radius + 6

            if self.intersects_reserved(x1, x2, y1, y2):
                continue

            for _ in range(points):
                angle = random.random() * 2 * math.pi
                dist = random.random() * radius

                x = int(cx + math.cos(angle) * dist)
                y = int(cy + math.sin(angle) * dist)

                rr = random.randint(2, 5)

                for dx in range(-rr, rr + 1):
                    for dy in range(-rr, rr + 1):
                        nx = x + dx
                        ny = y + dy

                        if 0 <= nx < self.h and 0 <= ny < self.w:
                            if dx * dx + dy * dy <= rr * rr:
                                grid[nx, ny] = OBSTACLE

            return

    def add_home_block(self, grid):
        """
        Добавляет прямоугольное препятствие-дом
        и возвращает стартовую позицию рядом с ним.
        """

        h, w = grid.shape

        block_h = max(4, h // 8)
        block_w = max(5, w // 10)

        x1 = h // 2
        y1 = w // 2

        x2 = min(h - 2, x1 + block_h)
        y2 = min(w - 2, y1 + block_w)

        grid[x1:x2, y1:y2] = -1

        margin = 6
        self.reserved_zones.append((
            max(0, x1 - margin),
            min(h, x2 + margin),
            max(0, y1 - margin),
            min(w, y2 + margin),
        ))

        start = (
            (x1 + x2) // 2,
            y2 + 1,
        )
        self.start_pos = start

        return start

    def intersects_reserved(self, x1, x2, y1, y2):
        for rx1, rx2, ry1, ry2 in self.reserved_zones:
            if not (
                    x2 <= rx1 or
                    x1 >= rx2 or
                    y2 <= ry1 or
                    y1 >= ry2
            ):
                return True

        return False