class AdaptiveTrafficController:
    """
    Динамически меняет параметры traffic/cell ordering
    по фазе миссии.

    Идея:
    - в начале миссии не боимся старых коридоров
    - в середине балансируем
    - в конце сильнее избегаем overlap
    """

    def __init__(self):
        self.phase = "EARLY"

    def update(self, env, runtime_config):
        coverage = env.env.coverage_rate()
        overlap = env.env.overlap_rate()
        energy_ratio = env.energy_system.energy / max(
            1e-9,
            env.energy_system.max_energy,
        )

        if coverage < 0.35:
            self.phase = "EARLY"
        elif coverage < 0.80:
            self.phase = "MID"
        else:
            self.phase = "LATE"

        if self.phase == "EARLY":
            runtime_config.set("VISIT_WEIGHT", 0.03)
            runtime_config.set("CELL_TRAFFIC_WEIGHT", 0.6)
            runtime_config.set("CUT_WEIGHT", 0.05)

        elif self.phase == "MID":
            runtime_config.set("VISIT_WEIGHT", 0.04)
            runtime_config.set("CELL_TRAFFIC_WEIGHT", 0.8)
            runtime_config.set("CUT_WEIGHT", 0.08)

        else:
            runtime_config.set("VISIT_WEIGHT", 0.06)
            runtime_config.set("CELL_TRAFFIC_WEIGHT", 1.0)
            runtime_config.set("CUT_WEIGHT", 0.10)

        # Если overlap уже высокий — усиливаем избегание горячих зон
        if overlap > 0.35:
            runtime_config.set("VISIT_WEIGHT", 0.08)
            runtime_config.set("CELL_TRAFFIC_WEIGHT", 1.2)

        # Если энергии мало — не делаем длинные detour
        if energy_ratio < 0.35:
            runtime_config.set("CELL_TRAFFIC_WEIGHT", 0.5)
            runtime_config.set("CUT_WEIGHT", 0.04)

        return {
            "adaptive_phase": self.phase,
            "adaptive_coverage": coverage,
            "adaptive_overlap": overlap,
            "adaptive_energy_ratio": energy_ratio,
            "adaptive_visit_weight": runtime_config.get("VISIT_WEIGHT"),
            "adaptive_cell_traffic_weight": runtime_config.get("CELL_TRAFFIC_WEIGHT"),
            "adaptive_cut_weight": runtime_config.get("CUT_WEIGHT"),
        }