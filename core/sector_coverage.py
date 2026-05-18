from core.world_memory import CABBAGE
import numpy as np
from core.config import DIRECTION_BIAS_WEIGHT, BACKTRACK_PENALTY
from core.tuning_config import runtime_config


class SectorCoveragePlanner:
    def __init__(self):
        self.cached_sector = None
        self.cached_targets = []
        self.target_index = 0
        self.cached_lines = []
        self.line_index = 0

    def reset(self):
        self.cached_sector = None
        self.cached_targets = []
        self.target_index = 0
        self.cached_lines = []
        self.line_index = 0
        self.target_index = 0

    def build_targets(self, memory, sector_manager, sector_id):
        x1, x2, y1, y2 = sector_manager.get_sector_bounds(
            sector_id,
            memory.map.shape
        )

        targets = []

        for x in range(x1, x2):
            cols = range(y1, y2)

            if (x - x1) % 2 == 1:
                cols = reversed(list(cols))

            for y in cols:
                if memory.map[x][y] == CABBAGE:
                    targets.append((x, y))

        return targets

    def get_next_target(self, memory, env, sector_manager, sector_id):
        if sector_id is None:
            return None

        if sector_id != self.cached_sector:
            self.cached_sector = sector_id
            self.cached_targets = self.build_targets(
                memory,
                sector_manager,
                sector_id
            )
            self.target_index = 0

        while self.target_index < len(self.cached_targets):
            x, y = self.cached_targets[self.target_index]

            if memory.map[x][y] == CABBAGE:
                return (x, y)

            self.target_index += 1

        self.cached_targets = self.build_targets(
            memory,
            sector_manager,
            sector_id
        )
        self.target_index = 0

        if len(self.cached_targets) == 0:
            return None

        return self.cached_targets[0]

    def get_next_target_directional(
            self,
            memory,
            env,
            sector_manager,
            sector_id,
            prev_pos=None
    ):
        if sector_id is None:
            return None

        # если сектор сменился — перестроить targets
        if sector_id != self.cached_sector:
            self.cached_sector = sector_id
            self.cached_targets = self.build_targets(
                memory,
                sector_manager,
                sector_id
            )
            self.target_index = 0

        # только ещё существующая известная капуста
        candidates = [
            p for p in self.cached_targets
            if memory.map[p[0], p[1]] == 1
        ]

        if len(candidates) == 0:
            self.cached_targets = self.build_targets(
                memory,
                sector_manager,
                sector_id
            )
            candidates = [
                p for p in self.cached_targets
                if memory.map[p[0], p[1]] == 1
            ]

        if len(candidates) == 0:
            return None

        x, y = env.pos

        # направление текущего движения
        if prev_pos is not None:
            px, py = prev_pos
            move_vec = np.array([x - px, y - py], dtype=np.float32)
        else:
            move_vec = np.array([0.0, 0.0], dtype=np.float32)

        best = None
        best_score = 1e18

        for tx, ty in candidates:
            dist = abs(tx - x) + abs(ty - y)

            target_vec = np.array([tx - x, ty - y], dtype=np.float32)

            direction_penalty = 0.0

            if np.linalg.norm(move_vec) > 0 and np.linalg.norm(target_vec) > 0:
                dot = np.dot(move_vec, target_vec)

                # если цель "назад" относительно текущего движения
                if dot < 0:
                    direction_penalty += BACKTRACK_PENALTY

                # если цель не по текущей оси движения
                cos_sim = dot / (
                        np.linalg.norm(move_vec) * np.linalg.norm(target_vec) + 1e-6
                )

                direction_penalty += DIRECTION_BIAS_WEIGHT * (1.0 - cos_sim)

            score = dist + direction_penalty

            if score < best_score:
                best_score = score
                best = (tx, ty)

        return best

    def build_sweep_lines(self, memory, sector_manager, sector_id):
        x1, x2, y1, y2 = sector_manager.get_sector_bounds(
            sector_id,
            memory.map.shape
        )

        lines = []

        for x in range(x1, x2):
            row = []

            cols = range(y1, y2)

            if (x - x1) % 2 == 1:
                cols = reversed(list(cols))

            for y in cols:
                if memory.map[x, y] == 1:
                    row.append((x, y))

            if len(row) > 0:
                lines.append(row)

        return lines

    def get_next_target_sweep_line(
            self,
            memory,
            env,
            sector_manager,
            sector_id
    ):
        if sector_id is None:
            return None

        if sector_id != self.cached_sector:
            self.cached_sector = sector_id
            self.cached_lines = self.build_sweep_lines(
                memory,
                sector_manager,
                sector_id
            )
            self.line_index = 0
            self.target_index = 0

        if not hasattr(self, "cached_lines"):
            self.cached_lines = []

        while self.line_index < len(self.cached_lines):
            line = self.cached_lines[self.line_index]

            while self.target_index < len(line):
                x, y = line[self.target_index]

                if memory.map[x, y] == 1:
                    return (x, y)

                self.target_index += 1

            self.line_index += 1
            self.target_index = 0

        self.cached_lines = self.build_sweep_lines(
            memory,
            sector_manager,
            sector_id
        )
        self.line_index = 0
        self.target_index = 0

        if len(self.cached_lines) == 0:
            return None

        if len(self.cached_lines[0]) == 0:
            return None

        return self.cached_lines[0][0]

    def get_next_target_hybrid(
            self,
            memory,
            env,
            sector_manager,
            sector_id,
            prev_pos=None
    ):
        # основная sweep-цель
        sweep_target = self.get_next_target_sweep_line(
            memory,
            env,
            sector_manager,
            sector_id
        )

        if sweep_target is None:
            return None

        x, y = env.pos
        sx, sy = sweep_target

        sweep_dist = abs(sx - x) + abs(sy - y)

        # собираем локальных кандидатов рядом
        candidates = []

        radius = runtime_config.get("LOCAL_TARGET_RADIUS", 3)

        x1, x2, y1, y2 = sector_manager.get_sector_bounds(
            sector_id,
            memory.map.shape
        )

        for i in range(max(x1, x - radius), min(x2, x + radius + 1)):
            for j in range(max(y1, y - radius), min(y2, y + radius + 1)):
                if memory.map[i, j] == 1:
                    candidates.append((i, j))

        if len(candidates) == 0:
            return sweep_target

        best = sweep_target
        best_score = sweep_dist

        local_bonus = runtime_config.get("LOCAL_TARGET_BONUS", 2.0)
        stickiness = runtime_config.get("SWEEP_STICKINESS", 1.5)

        for tx, ty in candidates:
            dist = abs(tx - x) + abs(ty - y)

            score = dist

            # если это не sweep target — даём штраф за отклонение
            if (tx, ty) != sweep_target:
                score += stickiness

            # если локальная цель сильно ближе — бонус
            if dist + local_bonus < sweep_dist:
                score -= local_bonus

            # штраф за немедленный backtrack
            if prev_pos is not None:
                px, py = prev_pos
                if (tx, ty) == (px, py):
                    score += 10.0

            if score < best_score:
                best_score = score
                best = (tx, ty)

        return best