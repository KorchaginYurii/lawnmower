import numpy as np


class LawnLaneMemory:
    """
    Память полос покоса.

    Работает с adapter grid:
        -1 = obstacle / buffer
         0 = empty
         1 = uncut grass
         2 = cut grass
    """

    def __init__(self, coverage_threshold=0.95):
        self.coverage_threshold = coverage_threshold
        self.orientation = "horizontal"

        self.completed_lanes = set()
        self.blocked_lanes = set()
        self.active_lane = None
        self.last_lane = None

    def reset(self):
        self.completed_lanes = set()
        self.blocked_lanes = set()
        self.active_lane = None
        self.last_lane = None

    def update(self, env):
        self.detect_orientation(env)

        lane_id = self.get_lane_id(env.pos)

        self.active_lane = lane_id
        self.last_lane = lane_id

        self.update_completed_lanes(env)

    def detect_orientation(self, env):
        h, w = env.grid.shape
        self.orientation = "horizontal" if w >= h else "vertical"

    def get_lane_id(self, pos):
        x, y = pos
        if self.orientation == "horizontal":
            return x
        return y

    def get_lane_cells(self, env, lane_id):
        if self.orientation == "horizontal":
            x = lane_id
            if x < 0 or x >= env.grid.shape[0]:
                return []
            return [(x, y) for y in range(env.grid.shape[1])]

        y = lane_id
        if y < 0 or y >= env.grid.shape[1]:
            return []
        return [(x, y) for x in range(env.grid.shape[0])]

    def lane_stats(self, env, lane_id):
        cells = self.get_lane_cells(env, lane_id)

        mowable = 0
        cut = 0
        uncut = 0
        blocked = 0

        for x, y in cells:
            v = env.grid[x, y]

            if v == -1:
                blocked += 1
            elif v == 1:
                mowable += 1
                uncut += 1
            elif v == 2:
                mowable += 1
                cut += 1

        coverage = cut / max(1, mowable)

        return {
            "mowable": mowable,
            "cut": cut,
            "uncut": uncut,
            "blocked": blocked,
            "coverage": coverage,
        }

    def update_completed_lanes(self, env):
        lane_count = (
            env.grid.shape[0]
            if self.orientation == "horizontal"
            else env.grid.shape[1]
        )

        for lane_id in range(lane_count):
            stats = self.lane_stats(env, lane_id)

            if stats["mowable"] == 0:
                self.blocked_lanes.add(lane_id)
                continue

            if stats["coverage"] >= self.coverage_threshold:
                self.completed_lanes.add(lane_id)

    def is_lane_completed(self, env, lane_id):
        if lane_id in self.completed_lanes:
            return True

        stats = self.lane_stats(env, lane_id)
        return stats["coverage"] >= self.coverage_threshold

    def next_unfinished_lane(self, env, from_lane=None):
        self.update_completed_lanes(env)

        lane_count = (
            env.grid.shape[0]
            if self.orientation == "horizontal"
            else env.grid.shape[1]
        )

        if from_lane is None:
            from_lane = self.active_lane
        if from_lane is None:
            from_lane = 0

        candidates = []

        for lane_id in range(lane_count):
            if lane_id in self.completed_lanes:
                continue
            if lane_id in self.blocked_lanes:
                continue

            stats = self.lane_stats(env, lane_id)

            if stats["uncut"] <= 0:
                continue

            dist = abs(lane_id - from_lane)

            score = (
                stats["uncut"] * 10.0
                - dist * 2.0
                - stats["blocked"] * 0.1
            )

            candidates.append((score, lane_id))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return candidates[0][1]

    def first_uncut_cell_in_lane(self, env, lane_id, prefer_pos=None):
        cells = self.get_lane_cells(env, lane_id)

        uncut = [
            (x, y)
            for x, y in cells
            if env.grid[x, y] == 1
        ]

        if not uncut:
            return None

        if prefer_pos is None:
            return uncut[0]

        px, py = prefer_pos
        uncut.sort(
            key=lambda p: abs(p[0] - px) + abs(p[1] - py)
        )

        return uncut[0]

    def progress_report(self, env):
        self.update_completed_lanes(env)

        lane_count = (
            env.grid.shape[0]
            if self.orientation == "horizontal"
            else env.grid.shape[1]
        )

        completed = len(self.completed_lanes)
        blocked = len(self.blocked_lanes)

        return {
            "orientation": self.orientation,
            "active_lane": self.active_lane,
            "completed_lanes": completed,
            "blocked_lanes": blocked,
            "total_lanes": lane_count,
            "lane_progress": completed / max(1, lane_count),
        }