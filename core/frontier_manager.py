import numpy as np
from collections import deque

from core.config import ACTIONS, MOVE_COST
from core.world_memory import UNKNOWN


class FrontierManager:
    def __init__(self):
        self.frontier_clusters = []
        self.selected_frontier = None

    def cluster_frontiers(self, memory):
        frontiers = set(memory.frontier_cells())
        clusters = []

        while frontiers:
            start = frontiers.pop()
            cluster = [start]
            q = deque([start])

            while q:
                x, y = q.popleft()

                for dx, dy in ACTIONS:
                    nx, ny = x + dx, y + dy
                    n = (nx, ny)

                    if n in frontiers:
                        frontiers.remove(n)
                        q.append(n)
                        cluster.append(n)

            clusters.append(cluster)

        self.frontier_clusters = clusters
        return clusters

    def cluster_center(self, cluster):
        arr = np.array(cluster)
        center = arr.mean(axis=0)
        return int(round(center[0])), int(round(center[1]))

    def unknown_gain_near_cluster(self, memory, cluster, radius=3):
        gain = 0
        h, w = memory.map.shape

        for x, y in cluster:
            for i in range(max(0, x - radius), min(h, x + radius + 1)):
                for j in range(max(0, y - radius), min(w, y + radius + 1)):
                    if memory.map[i, j] == UNKNOWN:
                        gain += 1

        return gain

    def choose_frontier(self, env, memory, planner, energy_predictor=None):
        clusters = self.cluster_frontiers(memory)

        best = None
        best_score = -1e9

        for cluster in clusters:
            center = self.cluster_center(cluster)

            path = planner.find_path_oriented(
                env,
                env.pos,
                center,
                memory=memory,
                unknown_policy="explore"
            )

            if path is None:
                continue

            path_home = planner.find_path_oriented(
                env,
                center,
                env.start_pos,
                memory=memory
            )

            if path_home is None:
                continue

            travel_cost = len(path) * MOVE_COST
            home_cost = len(path_home) * MOVE_COST

            if not env.energy_system.can_reach(travel_cost + home_cost, reserve=5.0):
                continue

            gain = self.unknown_gain_near_cluster(memory, cluster)
            size = len(cluster)

            score = gain * 1.5 + size * 2.0 - travel_cost - home_cost

            if score > best_score:
                best_score = score
                best = center

        self.selected_frontier = best
        return best