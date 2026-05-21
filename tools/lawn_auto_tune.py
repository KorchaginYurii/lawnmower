import os
import csv
import itertools
from datetime import datetime

import numpy as np

from agents.lawn_sweep_agent import LawnSweepAgent
from tools.benchmark_runner import (
    run_one_lawn_mission,
    BENCHMARK_SEEDS,
)

from core.config import (
    LAWN_PRESET,
    LAWN_PRESETS,
    CELL_SIZE_M,
    ROBOT_SIZE_M,
    LAWNMOWER_MAX_ENERGY,
)

from core.tuning_config import runtime_config


SEARCH_SPACE = {
    # traffic / hot corridor avoidance
    "VISIT_WEIGHT": [0.04, 0.06, 0.08, 0.10],
    #"CUT_WEIGHT": [0.02, 0.04, 0.06],
    #"TRAFFIC_MAX_PENALTY": [6.0, 8.0, 10.0],

    # coverage cell ordering
    "CELL_TRAFFIC_WEIGHT": [0.8, 1.0, 1.2, 1.5],
    "CELL_NEIGHBOR_BONUS": [15.0, 20.0, 25.0],
    #"CELL_DISTANCE_WEIGHT": [1.2, 1.5, 1.8],

    # energy
    "ENERGY_RESERVE": [25.0, 30.0, 35.0],
}


def config_product(space):
    keys = list(space.keys())
    values = [space[k] for k in keys]

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def score_results(rows):
    success_rate = np.mean([r["success"] for r in rows])
    coverage_rate = np.mean([r["coverage_rate"] for r in rows])
    overlap_rate = np.mean([r["overlap_rate"] for r in rows])
    energy_per_m2 = np.mean([r["energy_per_m2"] for r in rows])
    steps = np.mean([r["steps"] for r in rows])
    turns = np.mean([r["turns"] for r in rows])
    recharges = np.mean([r["recharges"] for r in rows])

    score = (
        success_rate * 3000.0
        + coverage_rate * 700.0
        - overlap_rate * 500.0
        - energy_per_m2 * 300.0
        - steps * 0.04
        - turns * 0.15
        - recharges * 12.0
    )

    metrics = {
        "success_rate": success_rate,
        "coverage_rate": coverage_rate,
        "overlap_rate": overlap_rate,
        "energy_per_m2": energy_per_m2,
        "avg_steps": steps,
        "avg_turns": turns,
        "avg_recharges": recharges,
    }

    return score, metrics


def save_tuning_results(rows, path=None):
    if path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"logs/lawn_auto_tune_{ts}.csv"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Auto-tune results saved: {path}")


def run_config(cfg, seeds):
    runtime_config.update(cfg)

    preset = LAWN_PRESETS[LAWN_PRESET]
    agent = LawnSweepAgent()

    rows = []

    for seed in seeds:
        result = run_one_lawn_mission(
            agent=agent,
            seed=seed,
            **preset,
            cell_size_m=CELL_SIZE_M,
            robot_size_m=ROBOT_SIZE_M,
            max_energy=LAWNMOWER_MAX_ENERGY,
        )
        rows.append(result)

    return rows


def main():
    all_rows = []

    best_score = -1e18
    best_config = None
    best_metrics = None

    configs = list(config_product(SEARCH_SPACE))

    print(f"Total configs: {len(configs)}")
    print(f"Seeds: {BENCHMARK_SEEDS}")

    for i, cfg in enumerate(configs):
        print("\n" + "=" * 70)
        print(f"CONFIG {i + 1}/{len(configs)}")
        print(cfg)

        rows = run_config(cfg, BENCHMARK_SEEDS)

        score, metrics = score_results(rows)

        print("\nscore:", round(score, 3))
        print("metrics:", metrics)

        flat_row = {
            "config_id": i,
            "score": score,
            **cfg,
            **metrics,
        }

        all_rows.append(flat_row)

        if score > best_score:
            best_score = score
            best_config = cfg.copy()
            best_metrics = metrics.copy()

            print("\n🔥 NEW BEST")
            print(best_config)
            print(best_metrics)

    save_tuning_results(all_rows)

    print("\n========== BEST CONFIG ==========")
    print(best_config)
    print("score:", best_score)
    print(best_metrics)


if __name__ == "__main__":
    main()