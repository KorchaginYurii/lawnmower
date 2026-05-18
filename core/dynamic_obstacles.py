import random
from core.config import ACTIONS


class DynamicObstacle:
    def __init__(self, pos):
        self.pos = pos
        self.prev_pos = pos

class DynamicObstacleManager:
    def __init__(self, count=2, move_prob=0.3):
        self.count = count
        self.move_prob = move_prob
        self.obstacles = []

    def reset(self, env):
        self.obstacles = []
        h, w = env.grid.shape
        free = [
            (i, j)
            for i in range(h)
            for j in range(w)
            if env.grid[i][j] == 0
            and (i, j) != env.pos
            and (i, j) != env.start_pos
        ]

        random.shuffle(free)

        for pos in free[:self.count]:
            self.obstacles.append(DynamicObstacle(pos))

    def positions(self):
        return {o.pos for o in self.obstacles}

    def step(self, env):
        occupied = self.positions()

        for obj in self.obstacles:
            if random.random() > self.move_prob:
                continue

            # запоминаем прошлую позицию перед движением
            obj.prev_pos = obj.pos

            x, y = obj.pos
            h, w = env.grid.shape

            moves = ACTIONS[:4].copy()
            random.shuffle(moves)

            for dx, dy in moves:
                nx = max(0, min(h - 1, x + dx))
                ny = max(0, min(w - 1, y + dy))
                np = (nx, ny)

                if np in env.obstacles:
                    continue

                if np in occupied:
                    continue

                if np == env.pos or np == env.start_pos:
                    continue

                if env.grid[nx][ny] == 1:
                    continue

                occupied.remove(obj.pos)

                obj.prev_pos = obj.pos
                obj.pos = np

                occupied.add(np)
                break

    def predicted_positions(self, horizon=3):
        preds = set()

        for obj in self.obstacles:
            x, y = obj.pos
            px, py = obj.prev_pos

            vx = x - px
            vy = y - py

            for t in range(1, horizon + 1):
                preds.add((x + vx * t, y + vy * t))

        return preds

    def predicted_positions(self, horizon=3):
        preds = {}

        for obj in self.obstacles:
            x, y = obj.pos
            px, py = obj.prev_pos

            vx = x - px
            vy = y - py

            if vx == 0 and vy == 0:
                continue

            for t in range(1, horizon + 1):
                p = (x + vx * t, y + vy * t)
                preds[p] = t

        return preds