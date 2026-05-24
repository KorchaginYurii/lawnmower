import csv
import os
from datetime import datetime

import pandas as pd

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


# =========================================================
# PROFILES
# =========================================================

PROFILES = [
    "configs.tuned_stable",
    "configs.tuned_aggressive",
    "configs.tuned_low_overlap",
    "configs.tuned_energy_saver",
]

ADAPTIVE_OPTIONS = [
    False,
    True,
]


# =========================================================
# RUN PROFILE
# =========================================================

def run_profile(profile_name, adaptive):

    runtime_config.reset()

    runtime_config.load_profile(profile_name)

    runtime_config.set(
        "USE_ADAPTIVE_TRAFFIC",
        adaptive,
    )

    preset = LAWN_PRESETS[LAWN_PRESET]

    agent = LawnSweepAgent()

    rows = []

    print("\n" + "=" * 80)
    print(f"PROFILE:  {profile_name}")
    print(f"ADAPTIVE: {adaptive}")
    print("=" * 80)

    for seed in BENCHMARK_SEEDS:

        print(f"\n🚜 seed={seed}")

        result = run_one_lawn_mission(
            agent=agent,
            seed=seed,

            **preset,

            cell_size_m=CELL_SIZE_M,
            robot_size_m=ROBOT_SIZE_M,
            max_energy=LAWNMOWER_MAX_ENERGY,
        )

        result["profile"] = runtime_config.get(
            "PROFILE_NAME",
            profile_name,
        )

        result["adaptive"] = adaptive

        rows.append(result)

    return rows


# =========================================================
# SUMMARY
# =========================================================

def summarize(rows):

    df = pd.DataFrame(rows)

    summary = {
        "profile":
            df["profile"].iloc[0],

        "adaptive":
            bool(df["adaptive"].iloc[0]),

        "success_rate":
            df["success"].mean(),

        "coverage_rate":
            df["coverage_rate"].mean(),

        "overlap_rate":
            df["overlap_rate"].mean(),

        "energy_per_m2":
            df["energy_per_m2"].mean(),

        "steps":
            df["steps"].mean(),

        "turns":
            df["turns"].mean(),

        "recharges":
            df["recharges"].mean(),
    }

    return summary


# =========================================================
# SAVE
# =========================================================

def save_summary(rows):

    ts = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    path = (
        f"logs/profile_benchmark_"
        f"{ts}.csv"
    )

    os.makedirs("logs", exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    print(f"\n✅ saved: {path}")


# =========================================================
# MAIN
# =========================================================

def main():

    all_summary_rows = []

    for profile in PROFILES:

        for adaptive in ADAPTIVE_OPTIONS:

            rows = run_profile(
                profile,
                adaptive,
            )

            summary = summarize(rows)

            all_summary_rows.append(
                summary
            )

            print("\nSUMMARY:")
            print(summary)

    save_summary(all_summary_rows)

    print("\n========== FINAL ==========")

    df = pd.DataFrame(all_summary_rows)

    print(
        df.sort_values(
            by=[
                "success_rate",
                "energy_per_m2",
                "overlap_rate",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        ).to_string(index=False)
    )

    print("===========================\n")


if __name__ == "__main__":
    main()