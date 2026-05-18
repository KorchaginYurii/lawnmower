import numpy as np
from core.world_memory import CABBAGE, OBSTACLE, UNKNOWN
from core.config import SECTOR_H, SECTOR_W

class SectorManager:
    def __init__(self, sector_h=SECTOR_H, sector_w=SECTOR_W):
        self.sector_h = sector_h
        self.sector_w = sector_w
        self.current_sector = None

    def get_sector_id(self, pos):
        x, y = pos
        return (x // self.sector_h, y // self.sector_w)

    def get_sector_bounds(self, sector_id, grid_shape):
        sx, sy = sector_id
        h, w = grid_shape

        x1 = sx * self.sector_h
        y1 = sy * self.sector_w

        x2 = min(x1 + self.sector_h, h)
        y2 = min(y1 + self.sector_w, w)

        return x1, x2, y1, y2

    def sector_cabbages(self, memory, sector_id):
        x1, x2, y1, y2 = self.get_sector_bounds(
            sector_id,
            memory.map.shape
        )

        sector = memory.map[x1:x2, y1:y2]

        return int(np.sum(sector == CABBAGE))

    def sector_center(self, sector_id, grid_shape):
        x1, x2, y1, y2 = self.get_sector_bounds(sector_id, grid_shape)
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def all_sectors(self, memory):
        h, w = memory.map.shape
        sectors = []

        for x in range(0, h, self.sector_h):
            for y in range(0, w, self.sector_w):
                sectors.append((x // self.sector_h, y // self.sector_w))

        return sectors

    def choose_sector(self, env, planner):
        # 1. Если текущий сектор ещё содержит капусту — остаёмся
        if self.current_sector is not None:
            current_count = self.sector_cabbages(env, self.current_sector)

            if current_count > 0:
                return self.current_sector
                # сектор пустой — сбрасываем

            self.current_sector = None

        # 2. Ищем новый сектор с капустой
        best_sector = None
        best_score = -1e9

        for sector_id in self.all_sectors(env):
            cab_count = self.sector_cabbages(env, sector_id)

            if cab_count == 0:
                continue

            center = self.sector_center(sector_id, env.grid.shape)
            path = planner.find_path(env, env.pos, center)

            if path is None:
                continue

            dist = len(path)

            # чем больше капусты и ближе сектор — тем лучше
            score = cab_count * 10 - dist

            if score > best_score:
                best_score = score
                best_sector = sector_id

        self.current_sector = best_sector
        return best_sector

    def nearest_cabbage_in_sector(self, memory, env, sector_id):
        if sector_id is None:
            return None

        x1, x2, y1, y2 = self.get_sector_bounds(
            sector_id,
            memory.map.shape
        )

        sector = memory.map[x1:x2, y1:y2]

        cabbages = np.argwhere(sector == CABBAGE)

        if len(cabbages) == 0:
            return None

        cabbages[:, 0] += x1
        cabbages[:, 1] += y1

        x, y = env.pos
        dists = np.abs(cabbages - np.array([x, y])).sum(axis=1)

        nearest = cabbages[np.argmin(dists)]
        return tuple(nearest)

    def choose_sector_energy_aware(
            self,
            env,
            memory,
            planner,
            energy_predictor,
            robot_id=None,
            blackboard=None
    ):
        best_sector = None
        best_score = -1e9
        best_required = None
        robot_id = None,
        blackboard = None

        # если текущий сектор ещё содержит известную капусту — остаёмся
        if self.current_sector is not None:
            current_count = self.sector_cabbages(memory, self.current_sector)

            if current_count > 0:
                return self.current_sector

            self.current_sector = None

        for sector_id in self.all_sectors(memory):
            cab_count = self.sector_cabbages(memory, sector_id)

            if cab_count == 0:
                continue
            if blackboard is not None and robot_id is not None:
                if not blackboard.is_sector_available(robot_id, sector_id):
                    continue

            center = self.sector_center(sector_id, memory.map.shape)

            path_to_sector = planner.find_path_oriented(
                env,
                env.pos,
                center
            )

            if path_to_sector is None:
                continue

            ok_energy, required_energy = energy_predictor.has_energy_to_finish_sector(
                env,
                planner,
                self,
                sector_id,
                memory=memory
            )

            if not ok_energy:
                continue

            travel_cost = len(path_to_sector) * 0.1

            score = cab_count * 10 - travel_cost - required_energy

            if score > best_score:
                best_score = score
                best_sector = sector_id
                best_required = required_energy

        self.current_sector = best_sector
        self.last_sector_required_energy = best_required

        return best_sector

    def all_sector_ids(self, memory):
        h, w = memory.map.shape

        ids = []

        for x in range(0, h, self.sector_h):
            for y in range(0, w, self.sector_w):
                ids.append((x // self.sector_h, y // self.sector_w))

        return ids
