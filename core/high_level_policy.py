import numpy as np


class HighLevelPolicy:
    """
    High-level policy поверх planner.

    Пока это rule-based baseline.
    Потом заменим на RL-модель.

    Action:
        0 = CONSERVATIVE
        1 = BALANCED
        2 = AGGRESSIVE
        3 = RETURN_HOME
        4 = RECOVERY
    """

    CONSERVATIVE = 0
    BALANCED = 1
    AGGRESSIVE = 2
    RETURN_HOME = 3
    RECOVERY = 4

    def __init__(self):
        self.last_action = self.BALANCED

    def reset(self):
        self.last_action = self.BALANCED

    def state(self, env, agent):
        coverage = env.env.coverage_rate()
        overlap = env.env.overlap_rate()

        energy = env.energy_system.energy
        max_energy = env.energy_system.max_energy
        energy_ratio = energy / max(1e-9, max_energy)

        remaining = env.env.remaining_grass()
        total = env.env.total_grass()

        remaining_ratio = remaining / max(1, total)

        return np.array([
            coverage,
            overlap,
            energy_ratio,
            remaining_ratio,
            agent.no_cut_counter / 50.0,
            agent.loop_counter / 10.0,
            agent.stuck_counter / 10.0,
        ], dtype=np.float32)

    def act(self, env, agent):
        s = self.state(env, agent)

        coverage = s[0]
        overlap = s[1]
        energy_ratio = s[2]
        remaining_ratio = s[3]
        no_cut = s[4]

        if energy_ratio < 0.25:
            action = self.RETURN_HOME

        elif no_cut > 0.5:
            action = self.RECOVERY

        elif coverage < 0.35:
            action = self.AGGRESSIVE

        elif overlap > 0.35:
            action = self.CONSERVATIVE

        elif remaining_ratio < 0.10:
            action = self.CONSERVATIVE

        else:
            action = self.BALANCED

        self.last_action = action
        return action

    def apply(self, action, runtime_config):
        if action == self.CONSERVATIVE:
            runtime_config.set("VISIT_WEIGHT", 0.07)
            runtime_config.set("CUT_WEIGHT", 0.10)
            runtime_config.set("CELL_TRAFFIC_WEIGHT", 1.1)

        elif action == self.BALANCED:
            runtime_config.set("VISIT_WEIGHT", 0.04)
            runtime_config.set("CUT_WEIGHT", 0.08)
            runtime_config.set("CELL_TRAFFIC_WEIGHT", 0.8)

        elif action == self.AGGRESSIVE:
            runtime_config.set("VISIT_WEIGHT", 0.03)
            runtime_config.set("CUT_WEIGHT", 0.06)
            runtime_config.set("CELL_TRAFFIC_WEIGHT", 0.6)

        elif action == self.RETURN_HOME:
            runtime_config.set("ENERGY_RESERVE", 35.0)

        elif action == self.RECOVERY:
            runtime_config.set("VISIT_WEIGHT", 0.08)
            runtime_config.set("CELL_TRAFFIC_WEIGHT", 1.2)

    def debug(self):
        names = {
            self.CONSERVATIVE: "CONSERVATIVE",
            self.BALANCED: "BALANCED",
            self.AGGRESSIVE: "AGGRESSIVE",
            self.RETURN_HOME: "RETURN_HOME",
            self.RECOVERY: "RECOVERY",
        }

        return {
            "hl_action": self.last_action,
            "hl_mode": names.get(self.last_action, "UNKNOWN"),
        }