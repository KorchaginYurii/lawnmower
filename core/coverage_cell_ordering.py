from core.tuning_config import runtime_config


class CoverageCellOrdering:
    """
    Оптимизатор порядка обхода coverage-cells.
    Параметры читаются из runtime_config во время score_cell().
    """

    def __init__(
        self,
        distance_weight=1.5,
        traffic_weight=1.2,
        neighbor_bonus=20.0,
        uncut_weight=3.0,
        return_home_weight=2.0,
    ):
        self.default_distance_weight = distance_weight
        self.default_traffic_weight = traffic_weight
        self.default_neighbor_bonus = neighbor_bonus
        self.default_uncut_weight = uncut_weight
        self.default_return_home_weight = return_home_weight

    @property
    def distance_weight(self):
        return runtime_config.get(
            "CELL_DISTANCE_WEIGHT",
            self.default_distance_weight,
        )

    @property
    def traffic_weight(self):
        return runtime_config.get(
            "CELL_TRAFFIC_WEIGHT",
            self.default_traffic_weight,
        )

    @property
    def neighbor_bonus(self):
        return runtime_config.get(
            "CELL_NEIGHBOR_BONUS",
            self.default_neighbor_bonus,
        )

    @property
    def uncut_weight(self):
        return runtime_config.get(
            "CELL_UNCUT_WEIGHT",
            self.default_uncut_weight,
        )

    @property
    def return_home_weight(self):
        return runtime_config.get(
            "CELL_RETURN_HOME_WEIGHT",
            self.default_return_home_weight,
        )

    def score_cell(
        self,
        env,
        decomposition,
        mission,
        traffic_cost,
        current_cell,
        candidate_cell,
        home_pos,
    ):
        cell = decomposition.cells[candidate_cell]

        center = cell.center()

        if center is None:
            return -1e18

        cx, cy = center
        px, py = env.pos
        hx, hy = home_pos

        # ==========================================
        # remaining work
        # ==========================================

        uncut = mission.cell_uncut(
            env,
            decomposition,
            candidate_cell,
        )

        if uncut <= 0:
            return -1e18

        # ==========================================
        # coverage
        # ==========================================

        coverage = mission.cell_coverage(
            env,
            decomposition,
            candidate_cell,
        )

        # ==========================================
        # distances
        # ==========================================

        dist_to_cell = (
            abs(cx - px)
            + abs(cy - py)
        )

        dist_home = (
            abs(cx - hx)
            + abs(cy - hy)
        )

        # ==========================================
        # traffic penalty
        # ==========================================

        traffic = 0.0

        if hasattr(env.env, "visit_count"):
            traffic = (
                env.env.visit_count[cx, cy]
                * traffic_cost.visit_weight
            )

        # ==========================================
        # neighbor preference
        # ==========================================

        neighbor_bonus = 0.0

        if current_cell is not None:
            if (
                candidate_cell
                in decomposition.cells[
                    current_cell
                ].neighbors
            ):
                neighbor_bonus = self.neighbor_bonus

        # ==========================================
        # score
        # ==========================================

        score = (
            uncut * self.uncut_weight
            - coverage * 20.0
            - dist_to_cell * self.distance_weight
            - traffic * self.traffic_weight
            - dist_home * self.return_home_weight * 0.05
            + neighbor_bonus
        )

        return score

    def choose_best_cell(
        self,
        env,
        decomposition,
        mission,
        traffic_cost,
        candidate_cells,
        current_cell,
        home_pos,
    ):
        best = None
        best_score = -1e18

        for cell_id in candidate_cells:
            score = self.score_cell(
                env,
                decomposition,
                mission,
                traffic_cost,
                current_cell,
                cell_id,
                home_pos,
            )

            if score > best_score:
                best_score = score
                best = cell_id

        return best