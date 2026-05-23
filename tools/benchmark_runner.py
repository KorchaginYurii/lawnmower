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
    WAIT_ACTION,
)
from core.tuning_config import runtime_config
from core.config import USE_ADAPTIVE_TRAFFIC


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
    max_object_size=24,

    max_steps_mult=3,
    border_margin=1,

    min_total_reward=-5000.0,
    min_recent_reward=-1000.0,
    recent_window=300,
):



    # =====================================================
    # SEED
    # =====================================================

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # =====================================================
    # ENV
    # =====================================================

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
        max_object_size=max_object_size,
    )

    env = LawnHybridAdapter(lawn)

    # =====================================================
    # AGENT
    # =====================================================

    if hasattr(agent, "reset"):
        agent.reset()

    # =====================================================
    # LIMITS
    # =====================================================

    total_grass = lawn.total_grass()

    max_steps = max(
        500,
        int(total_grass * max_steps_mult)
    )

    # =====================================================
    # METRICS
    # =====================================================

    total_reward = 0.0

    reward_history = []
    recent_rewards = []

    fail_reason_override = None

    # =====================================================
    # MAIN LOOP
    # =====================================================

    for step in range(max_steps):

        env.sync_from_env()

        action, debug = agent.act(env)

        reward, done = env.step(action)

        total_reward += reward

        reward_history.append(reward)

        # =================================================
        # RECENT WINDOW
        # =================================================

        recent_rewards.append(reward)

        if len(recent_rewards) > recent_window:
            recent_rewards.pop(0)

        # =================================================
        # EARLY STOP: VERY BAD TOTAL REWARD
        # =================================================

        if total_reward <= min_total_reward:
            fail_reason_override = "reward_too_negative"
            break

        # =================================================
        # EARLY STOP: BAD RECENT REWARD
        # =================================================

        near_complete = lawn.remaining_grass() <= 3

        if (
            not near_complete
            and len(recent_rewards) >= recent_window
            and sum(recent_rewards) <= min_recent_reward
        ):
            fail_reason_override = "recent_reward_too_negative"
            break

        # =================================================
        # STUCK ON BASE
        # =================================================

        if (
            lawn.pos == lawn.start_pos
            and lawn.remaining_grass() > 5
            and step > 20
            and action == WAIT_ACTION
        ):
            fail_reason_override = "stuck_on_base"
            break

        # =================================================
        # TOO MANY RECHARGES
        # =================================================

        if getattr(lawn, "recharge_count", 0) > 30:
            fail_reason_override = "too_many_recharges"
            break

        # =================================================
        # SPIN LOOP DETECTION
        # =================================================

        if lawn.steps > 300:

            turn_ratio = (
                lawn.total_turns
                / max(1, lawn.steps)
            )

            if (
                turn_ratio > 0.95
                and lawn.coverage_rate() < 0.2
            ):
                fail_reason_override = "spin_loop"
                break

        # =================================================
        # NORMAL DONE
        # =================================================
        ###############
            st_bar = int(20 * (step + 1) / max_steps)
            print(
                    f"*\r{seed}* "
                    f"STEP [{'#' * st_bar}{'-' * (20 - st_bar)}] {step + 1}/{max_steps} | "
                    f"recharge= {lawn.recharge_count} | "
                    f"grass= {lawn.remaining_grass()}/{total_grass} | ",
                    f"reward= {total_reward:2f}",
                    end="",
                    flush=True
             )
        ############
        if done:
            break
    # =====================================================
    # FINAL METRICS
    # =====================================================

    remaining = lawn.remaining_grass()

    cut = lawn.cut_grass()

    total_grass = lawn.total_grass()

    coverage_rate = lawn.coverage_rate()

    overlap_rate = lawn.overlap_rate()

    all_cut = remaining == 0

    at_start = lawn.pos == lawn.start_pos

    near_complete = (
        coverage_rate >= 0.999
        and remaining <= 3
    )

    success = int(
        (all_cut and at_start)
        or (near_complete and at_start)
    )

    # =====================================================
    # FAIL REASON
    # =====================================================

    if success:

        if all_cut:
            fail_reason = "success"
        else:
            fail_reason = "near_complete_success"

    elif fail_reason_override is not None:

        fail_reason = fail_reason_override

    elif all_cut and not at_start:

        fail_reason = "no_return"

    elif lawn.energy <= 0:

        fail_reason = "energy_empty"

    else:

        fail_reason = "not_finished"

    # =====================================================
    # EFFICIENCY
    # =====================================================

    if cut > 0:
        energy_per_m2 = lawn.energy_used / cut
    else:
        energy_per_m2 = 999999.0

    # =====================================================
    # RESULT
    # =====================================================

    result = {
        "profile": runtime_config.get(
            "PROFILE_NAME",
            "unknown"
        ),
        "adaptive": USE_ADAPTIVE_TRAFFIC,

        "adaptive_phase": debug.get("adaptive_phase", "NA"),
        "visit_weight": runtime_config.get("VISIT_WEIGHT"),
        "cell_traffic_weight": runtime_config.get("CELL_TRAFFIC_WEIGHT"),
        "cut_weight": runtime_config.get("CUT_WEIGHT"),




        # =========================
        # BASIC
        # =========================

        "seed": seed,

        "success": success,

        "fail_reason": fail_reason,

        # =========================
        # COVERAGE
        # =========================

        "cut_cells": cut,

        "remaining_grass": remaining,

        "total_grass": total_grass,

        "coverage_rate": coverage_rate,

        "overlap_rate": overlap_rate,

        # =========================
        # ENERGY
        # =========================

        "energy_used": lawn.energy_used,

        "energy_left": lawn.energy,

        "energy_percent_left": (
            lawn.energy
            / max(1e-9, lawn.max_energy)
            * 100.0
        ),

        "energy_per_m2": energy_per_m2,

        "recharges": getattr(
            lawn,
            "recharge_count",
            0,
        ),

        # =========================
        # MOVEMENT
        # =========================

        "steps": lawn.steps,

        "limit": max_steps,

        "turns": lawn.total_turns,

        # =========================
        # REWARD
        # =========================

        "total_reward": total_reward,

        "avg_reward": (
            total_reward
            / max(1, lawn.steps)
        ),

        # =========================
        # POSITION
        # =========================

        "final_pos": lawn.pos,

        "at_start": at_start,

        # =========================
        # MAP
        # =========================

        "map_h": lawn.h,

        "map_w": lawn.w,

        "object_count": object_count,

        "max_object_size": max_object_size,

        "border_margin": border_margin,
    }

    return result
def save_results(
    rows,
    path=None,
):
    if path is None:
        path = f"logs/benchmark_{LAWNMOWER_BENCHMARK_VERSION}_{ts}.csv"

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
    runtime_config.reset()

    runtime_config.load_profile(
        "configs.tuned_aggressive"
    )
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