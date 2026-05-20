import os
import csv
import random
from datetime import datetime

import numpy as np
import torch

from adapters.lawn_hybrid_adapter import LawnHybridAdapter
from env.lawn_env import LawnEnv

from agents.hybrid_agent import HybridAgent
from agents.lawn_sweep_agent import LawnSweepAgent
from core.config import (
    LAWN_PRESET,
    LAWN_PRESETS,
    CELL_SIZE_M,
    ROBOT_SIZE_M,
    LAWNMOWER_MAX_ENERGY,
)

LAWNMOWER_BENCHMARK_VERSION = "lawn_sweep_v1"

BENCHMARK_SEEDS = [
    101, 102, 103, 104, 105,
    201, 202, 203, 204, 205,
]

ts = datetime.now().strftime("%Y%m%d_%H%M")


def run_one_lawn_mission(
    agent,
    seed,
    width_m=42,
    height_m=45,
    cell_size_m=0.25,
    robot_size_m=0.5,
    max_energy=100.0,
    object_count=10,
    max_steps_mult=3,
    border_margin=1,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    lawn = LawnEnv(
        width_m=width_m,
        height_m=height_m,
        cell_size_m=cell_size_m,
        robot_size_m=robot_size_m,
        max_energy=max_energy,
    )

    lawn.reset_realistic_lawn(
        object_count=object_count,
        seed=seed,
        border_margin=border_margin,
    )

    env = LawnHybridAdapter(lawn)

    if hasattr(agent, "reset"):
        agent.reset()

    total_reward = 0.0
    debug = {}

    total_grass = lawn.total_grass()

    episode_limit = int(total_grass * max_steps_mult)

    for step in range(episode_limit):
        env.sync_from_env()

        action, debug = agent.act(env, temp=0)

        reward, done = env.step(action)

        ###############
        st_bar = int(20 * (step + 1) / episode_limit)
        print(
            f"\rSTEP [{'#' * st_bar}{'-' * (20 - st_bar)}] {step + 1}/{episode_limit} | "
            f"pos= {lawn.pos} | "
            f"cell= {lawn.grid[lawn.pos]}",
            end="",
            flush=True
        )
        ############
        total_reward += reward
        if done:
            break

    remaining = lawn.remaining_grass()
    cut = lawn.cut_grass()

    coverage_rate = lawn.coverage_rate()
    overlap_rate = lawn.overlap_rate()

    all_cut = remaining == 0
    at_start = lawn.pos == lawn.start_pos

    success = int(all_cut and at_start)

    if success:
        fail_reason = "success"
    elif all_cut and not at_start:
        fail_reason = "no_return"
    elif lawn.energy <= 0:
        fail_reason = "energy_empty"
    else:
        fail_reason = "not_finished"

    energy_used = lawn.energy_used
    energy_percent_left = lawn.energy / lawn.max_energy * 100.0

    area_m2 = cut * (cell_size_m ** 2)

    energy_per_m2 = energy_used / max(1e-9, area_m2)

    recovery_counts = debug.get("recovery_counts", {})

    return {
        "seed": seed,
        "success": success,
        "fail_reason": fail_reason,

        "reward": total_reward,
        "steps": lawn.steps,
        "limit": episode_limit,

        "cut_cells": cut,
        "total_grass_cells": total_grass,
        "remaining_grass": remaining,

        "coverage_rate": coverage_rate,
        "overlap_rate": overlap_rate,

        "energy_used": energy_used,
        "energy_left": lawn.energy,
        "energy_percent_left": energy_percent_left,
        "energy_per_m2": energy_per_m2,

        "turns": lawn.total_turns,

        "sector_switches": debug.get("sector_switches", 0),

        "recovery_waits": recovery_counts.get("WAIT", 0),
        "recovery_backoffs": recovery_counts.get("BACK_OFF", 0),
        "recovery_explore_alt": recovery_counts.get("EXPLORE_ALT", 0),

        "no_path_events": debug.get("no_path_counter", 0),
        "blocked_events": debug.get("blocked_counter", 0),

        "object_count": object_count,
        "cell_size_m": cell_size_m,
        "robot_size_m": robot_size_m,
        "max_energy": max_energy,
        "recharges": getattr(lawn, "recharge_count", 0),
    }


def save_results(
    rows,
    path=None,
):
    if path is None:
        path = f"logs/lawn_benchmark_{LAWNMOWER_BENCHMARK_VERSION}_{ts}.csv"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Lawn benchmark saved: {path}")


def print_summary(rows):
    success_rate = np.mean([r["success"] for r in rows])
    coverage = np.mean([r["coverage_rate"] for r in rows])
    energy = np.mean([r["energy_used"] for r in rows])
    energy_per_m2 = np.mean([r["energy_per_m2"] for r in rows])
    turns = np.mean([r["turns"] for r in rows])
    overlap = np.mean([r["overlap_rate"] for r in rows])
    steps = np.mean([r["steps"] for r in rows])

    print("\n========== LAWN BENCHMARK SUMMARY ==========")
    print(f"missions:          {len(rows)}")
    print(f"success_rate:      {success_rate:.2f}")
    print(f"coverage_rate:     {coverage:.3f}")
    print(f"avg_energy_used:   {energy:.2f}")
    print(f"energy_per_m2:     {energy_per_m2:.3f}")
    print(f"avg_steps:         {steps:.1f}")
    print(f"avg_turns:         {turns:.1f}")
    print(f"avg_overlap:       {overlap:.3f}")
    print("===========================================")


def main():
    agent = LawnSweepAgent()
    preset = LAWN_PRESETS[LAWN_PRESET]
    rows = []

    for i, seed in enumerate(BENCHMARK_SEEDS):
        print(f"\n🚜 Lawn benchmark {i + 1}/{len(BENCHMARK_SEEDS)} | seed={seed}")

        result = run_one_lawn_mission(
            agent=agent,
            seed=seed,

            **preset,

            cell_size_m=CELL_SIZE_M,
            robot_size_m=ROBOT_SIZE_M,

            max_energy=LAWNMOWER_MAX_ENERGY,
        )
        rows.append(result)

        print(
            f"\nseed={seed} | "
            f"success={result['success']} | "
            f"{result['fail_reason']} | "
            f"coverage={result['coverage_rate']:.3f} | "
            f"energy={result['energy_used']:.2f} | "
            f"left={result['energy_percent_left']:.1f}% | "
            f"steps={result['steps']}/{result['limit']} | "
            f"overlap={result['overlap_rate']:.3f}"
        )

    save_results(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()