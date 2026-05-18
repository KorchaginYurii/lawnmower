import numpy as np

from core.config import CUT_COST, MOVE_COST


class EnergyPredictor:
    def __init__(self, reserve=5.0):
        self.reserve = reserve

    def sector_remaining_cabbages(self, env, sector_manager, sector_id, memory=None):
        if sector_id is None:
            return 0

        if memory is not None:
            return sector_manager.sector_cabbages(memory, sector_id)

        # fallback по env.grid, если memory не передали
        x1, x2, y1, y2 = sector_manager.get_sector_bounds(
            sector_id,
            env.grid.shape
        )

        sector = env.grid[x1:x2, y1:y2]

        return int(np.sum(sector == 1))

    def estimate_sector_work_cost(self, env, sector_manager, sector_id, memory=None):
        """
        Грубая оценка стоимости дочистки сектора:
        - стоимость ножа на оставшуюся капусту
        - примерная стоимость движения внутри сектора
        """
        cab_count = self.sector_remaining_cabbages(
            env,
            sector_manager,
            sector_id,
            memory=memory
        )

        if cab_count == 0:
            return 0.0

        sector_h = sector_manager.sector_h
        sector_w = sector_manager.sector_w

        sweep_cost = (sector_h * sector_w) * 0.25 * MOVE_COST
        cut_cost = cab_count * CUT_COST

        return sweep_cost + cut_cost

    def estimate_safe_finish_cost(self, env, planner, sector_manager, sector_id, memory=None):
        """
        Сколько нужно энергии:
        1. доработать сектор
        2. вернуться домой
        3. оставить reserve
        """
        work_cost = self.estimate_sector_work_cost(
            env,
            sector_manager,
            sector_id,
            memory=memory
        )

        path_home = planner.find_path_oriented(
            env,
            env.pos,
            env.start_pos
        )

        if path_home is None:
            return float("inf")

        path_home_len_cost = len(path_home) * MOVE_COST

        return work_cost + path_home_len_cost + self.reserve

    def has_energy_to_finish_sector(self, env, planner, sector_manager, sector_id, memory=None):
        required = self.estimate_safe_finish_cost(
            env,
            planner,
            sector_manager,
            sector_id,
            memory=memory
        )

        return env.energy_system.energy >= required, required