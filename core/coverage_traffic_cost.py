import numpy as np


class CoverageTrafficCost:
    """
    Штраф за повторное использование одних и тех же клеток.

    Цель:
    - уменьшить горячие зоны
    - снизить overlap
    - заставить A* выбирать менее истоптанные коридоры
    """

    def __init__(
        self,
        visit_weight=0.08,
        cut_weight=0.05,
        max_penalty=8.0,
    ):
        self.visit_weight = visit_weight
        self.cut_weight = cut_weight
        self.max_penalty = max_penalty

    def cell_cost(self, env, pos):
        x, y = pos

        cost = 0.0

        if hasattr(env.env, "visit_count"):
            cost += env.env.visit_count[x, y] * self.visit_weight

        if hasattr(env.env, "cut_count"):
            cost += env.env.cut_count[x, y] * self.cut_weight

        return min(cost, self.max_penalty)

    def path_cost(self, env, path):
        if path is None:
            return 1e18

        return sum(
            self.cell_cost(env, p)
            for p in path
        )

    def best_path(
        self,
        env,
        planner,
        targets,
        max_targets=30,
    ):
        """
        Выбирает достижимую цель с минимальной ценой:
        distance + traffic penalty.
        """

        if not targets:
            return None, None

        scored_targets = []

        x, y = env.pos

        for t in targets:
            tx, ty = t
            dist = abs(tx - x) + abs(ty - y)
            scored_targets.append((dist, t))

        scored_targets.sort(key=lambda z: z[0])

        best_target = None
        best_path = None
        best_score = 1e18

        for _, target in scored_targets[:max_targets]:
            path = planner.find_path_oriented(
                env,
                env.pos,
                target,
                memory=None,
                unknown_policy="allow",
                robot_id="lawnmower",
                blackboard=None,
            )

            if path is None or len(path) < 2:
                continue

            score = len(path) + self.path_cost(env, path)

            if score < best_score:
                best_score = score
                best_target = target
                best_path = path

        return best_target, best_path