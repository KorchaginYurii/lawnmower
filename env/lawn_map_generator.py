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
        self.min_object_size = min_object_size
        self.max_object_size = max_object_size

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

    def add_circle(self, grid):
        r = random.randint(
            self.min_object_size,
            self.max_object_size // 2,
        )

        cx = random.randint(r + 2, self.h - r - 3)
        cy = random.randint(r + 2, self.w - r - 3)

        for x in range(cx - r, cx + r + 1):
            for y in range(cy - r, cy + r + 1):
                if 0 <= x < self.h and 0 <= y < self.w:
                    if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                        grid[x, y] = OBSTACLE

    def add_ellipse(self, grid):
        rx = random.randint(
            self.min_object_size,
            self.max_object_size,
        )
        ry = random.randint(
            self.min_object_size,
            self.max_object_size,
        )

        cx = random.randint(rx + 2, self.h - rx - 3)
        cy = random.randint(ry + 2, self.w - ry - 3)

        for x in range(cx - rx, cx + rx + 1):
            for y in range(cy - ry, cy + ry + 1):
                if 0 <= x < self.h and 0 <= y < self.w:
                    v = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
                    if v <= 1.0:
                        grid[x, y] = OBSTACLE

    def add_rect(self, grid):
        hh = random.randint(
            self.min_object_size,
            self.max_object_size,
        )
        ww = random.randint(
            self.min_object_size,
            self.max_object_size,
        )

        x1 = random.randint(2, self.h - hh - 3)
        y1 = random.randint(2, self.w - ww - 3)

        grid[x1:x1 + hh, y1:y1 + ww] = OBSTACLE

    def add_blob(self, grid):
        cx = random.randint(6, self.h - 7)
        cy = random.randint(6, self.w - 7)

        points = random.randint(12, 30)
        radius = random.randint(
            self.min_object_size,
            self.max_object_size,
        )

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