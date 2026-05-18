import os
import csv
import random
import numpy as np
import torch

from core.config import USE_LOCAL_RL
from core.config import MAP_H, MAP_W
from env.cabbage_env import CabbageEnv
from agents.cabbage_agent import CabbageAgent
from agents.hybrid_agent import HybridAgent
from core.checkpoint import CheckpointManager
from core.config import CONFIG_VERSION
from datetime import datetime

BENCHMARK_SEEDS = [
    101, 102, 103, 104, 105,
    201, 202, 203, 204, 205,
]
ts = datetime.now().strftime("%Y%m%d_%H%M")

def run_one_mission(agent, seed, max_extra_steps_mult=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = CabbageEnv(MAP_H, MAP_W)

    env.reset(
        obs_min=0.10,
        obs_max=0.20,
        cab_min=0.40,
        cab_max=0.55,
        seed=seed
    )

    if hasattr(agent, "reset"):
        agent.reset()

    total_reward = 0.0
    debug = {}

    episode_limit = env.max_steps + max(MAP_H, MAP_W) * max_extra_steps_mult

    for step in range(episode_limit):
        action, debug = agent.act(env, temp=0)
        reward, done = env.step(action)

        total_reward += reward

        if done:
            break

    total_cabbages = int(np.sum(env.initial_grid == 1))
    remaining = int(np.sum(env.grid == 1))
    collected = total_cabbages - remaining

    all_collected = remaining == 0
    at_start = env.pos == env.start_pos
    success = int(all_collected and at_start)

    if success:
        fail_reason = "success"
    elif all_collected and not at_start:
        fail_reason = "no_return"
    else:
        fail_reason = "not_collected"

    energy_used = getattr(env, "energy_used", 0.0)
    turns = getattr(env, "total_turns", 0)

    overlap_rate = debug.get("overlap_rate", 0.0)
    sector_switches = debug.get("sector_switches", 0)
    recharges = getattr(env, "recharge_count", 0)
    recovery_counts = debug.get("recovery_counts", {})

    return {
        "seed": seed,
        "success": success,
        "fail_reason": fail_reason,
        "reward": total_reward,
        "steps": env.steps,
        "limit": episode_limit,
        "collected": collected,
        "total_cabbages": total_cabbages,
        "collect_rate": collected / max(1, total_cabbages),
        "energy_used": energy_used,
        "energy_per_cabbage": energy_used / max(1, collected),
        "turns": turns,
        "overlap_rate": overlap_rate,
        "sector_switches": sector_switches,
        "recharges": recharges,
        "recovery_waits": recovery_counts.get("WAIT", 0),
        "recovery_backoffs": recovery_counts.get("BACK_OFF", 0),
        "recovery_explore_alt": recovery_counts.get("EXPLORE_ALT", 0),
        "no_path_events": debug.get("no_path_counter", 0),
        "blocked_events": debug.get("blocked_counter", 0),
    }


def save_results(rows, path=(f"logs/benchmark_{CONFIG_VERSION}_{ts}.csv")):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Benchmark saved: {path}")


def print_summary(rows):
    success_rate = np.mean([r["success"] for r in rows])
    collect_rate = np.mean([r["collect_rate"] for r in rows])
    energy = np.mean([r["energy_used"] for r in rows])
    e_per_cab = np.mean([r["energy_per_cabbage"] for r in rows])
    turns = np.mean([r["turns"] for r in rows])
    overlap = np.mean([r["overlap_rate"] for r in rows])

    print("\n========== BENCHMARK SUMMARY ==========")
    print(f"missions:           {len(rows)}")
    print(f"success_rate:       {success_rate:.2f}")
    print(f"collect_rate:       {collect_rate:.2f}")
    print(f"avg_energy_used:    {energy:.2f}")
    print(f"energy_per_cabbage: {e_per_cab:.3f}")
    print(f"avg_turns:          {turns:.1f}")
    print(f"avg_overlap:        {overlap:.2f}")
    print("=======================================")


def main():
    if USE_LOCAL_RL:
        local_agent = CabbageAgent()
        ckpt = CheckpointManager(
            k_best=3,
            project_name="Cab4"
        )
        ckpt.load_checkpoint(local_agent)

    agent = HybridAgent(
        local_agent=local_agent if USE_LOCAL_RL else None,
        robot_id="robot_1"
    )

    rows = []

    for i, seed in enumerate(BENCHMARK_SEEDS):#[:5]):
        print(f"\n🚀 Benchmark {i + 1}/{len(BENCHMARK_SEEDS)} | seed={seed}")

        result = run_one_mission(agent, seed)
        rows.append(result)

        print(
            f"seed={seed} | "
            f"success={result['success']} | "
            f"{result['fail_reason']} | "
            f"cabbage={result['collected']}/{result['total_cabbages']} | "
            f"energy={result['energy_used']:.1f} | "
            f"steps={result['steps']}/{result['limit']} | "
            f"recharges={result['recharges']} "
        )

    save_results(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()