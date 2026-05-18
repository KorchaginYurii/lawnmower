import numpy as np


class MissionPlanner:
    def __init__(self):
        self.route = []
        self.route_index = 0
        self.cached_signature = None

    def reset(self):
        self.route = []
        self.route_index = 0
        self.cached_signature = None

    def sector_center(self, sector_manager, sector_id, memory):
        x1, x2, y1, y2 = sector_manager.get_sector_bounds(
            sector_id,
            memory.map.shape
        )

        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )

    def build_signature(self, memory):
        return int(np.sum(memory.map == 1))

    def build_route(self, env, memory, sector_manager):
        self.route = []
        self.route_index = 0

        current = env.pos
        sectors = []

        for sector_id in sector_manager.all_sector_ids(memory):
            cabbages = sector_manager.sector_cabbages(memory, sector_id)

            if cabbages <= 0:
                continue

            center = self.sector_center(
                sector_manager,
                sector_id,
                memory
            )

            sectors.append({
                "id": sector_id,
                "center": center,
                "cabbages": cabbages,
            })

        while len(sectors) > 0:
            best_i = None
            best_score = -1e18

            cx, cy = current

            for i, s in enumerate(sectors):
                sx, sy = s["center"]

                dist = abs(sx - cx) + abs(sy - cy)

                # главный критерий:
                # много капусты и близко = хорошо
                score = s["cabbages"] / max(1, dist)

                # небольшой бонус плотным секторам
                score += 0.02 * s["cabbages"]

                if score > best_score:
                    best_score = score
                    best_i = i

            selected = sectors.pop(best_i)

            self.route.append(selected["id"])
            current = selected["center"]

        self.cached_signature = self.build_signature(memory)

    def current_sector(self, env, memory, sector_manager):
        need_rebuild = False

        if len(self.route) == 0:
            need_rebuild = True

        elif self.route_index >= len(self.route):
            need_rebuild = True

        elif sector_manager.current_sector is None:
            need_rebuild = True

        else:
            current = sector_manager.current_sector

            remaining = sector_manager.sector_cabbages(
                memory,
                current
            )

            if remaining == 0:
                need_rebuild = True

        if need_rebuild:
            self.build_route(
                env,
                memory,
                sector_manager
            )

        while self.route_index < len(self.route):
            sector_id = self.route[self.route_index]

            remaining = sector_manager.sector_cabbages(
                memory,
                sector_id
            )

            if remaining > 0:
                return sector_id

            self.route_index += 1

        return None