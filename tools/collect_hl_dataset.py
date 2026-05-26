import os
import pickle
import random

import numpy as np
import torch

from env.lawn_env import LawnEnv
from adapters.lawn_hybrid_adapter import LawnHybridAdapter

from agents.lawn_sweep_agent import LawnSweepAgent

from core.high_level_policy import HighLevelPolicy
from core.tuning_config import runtime_config

from core.config import (
    LAWN_PRESET,
    LAWN_PRESETS,
    CELL_SIZE_M,
    ROBOT_SIZE_M,
    LAWNMOWER_MAX_ENERGY,
)


SAVE_PATH = "datasets/high_level_policy_dataset.pkl"

EPISODES = 200


def make_env(seed):
    preset = LAWN_PRESETS[LAWN_PRESET]

    lawn = LawnEnv(
        width_m=preset["width_m"],
        height_m=preset["height_m"],
        cell_size_m=CELL_SIZE_M,
        robot_size_m=ROBOT_SIZE_M,
        max_energy=LAWNMOWER_MAX_ENERGY,
    )

    lawn.reset_realistic_lawn(
        object_count=preset["object_count"],
        seed=seed,
        border_margin=preset["border_margin"],
        max_object_size=preset["max_object_size"],
    )

    return lawn, LawnHybridAdapter(lawn)


def collect():
    runtime_config.reset()

    runtime_config.load_profile(
        "configs.tuned_stable"
    )

    runtime_config.set(
        "USE_ADAPTIVE_TRAFFIC",
        False,
    )

    runtime_config.set(
        "USE_HIGH_LEVEL_POLICY",
        True,
    )

    runtime_config.set(
        "USE_RL_HIGH_LEVEL_POLICY",
        False,
    )

    dataset = []

    teacher = HighLevelPolicy()

    for ep in range(EPISODES):

        seed = 1000 + ep

        lawn, env = make_env(seed)

        agent = LawnSweepAgent()

        max_steps = max(
            500,
            int(lawn.total_grass() * 3),
        )

        print(
            f"\nEP {ep} | "
            f"seed={seed}"
        )

        for step in range(max_steps):

            env.sync_from_env()

            state = teacher.state(
                env,
                agent,
            )

            action_hl = teacher.act(
                env,
                agent,
            )

            dataset.append(
                (
                    state.astype(np.float32),
                    int(action_hl),
                )
            )

            teacher.apply(
                action_hl,
                runtime_config,
            )

            action, debug = agent.act(env)

            reward, done = env.step(action)

            if step % 200 == 0:
                print(
                    f"step={step} | "
                    f"hl={debug.get('hl_mode')} | "
                    f"coverage={lawn.coverage_rate():.3f}"
                )

            if done:
                break

        print(
            f"dataset size = {len(dataset)}"
        )

    os.makedirs(
        os.path.dirname(SAVE_PATH),
        exist_ok=True,
    )

    with open(SAVE_PATH, "wb") as f:
        pickle.dump(dataset, f)

    print(
        f"\n✅ dataset saved: "
        f"{SAVE_PATH}"
    )

    print(
        f"samples: {len(dataset)}"
    )


if __name__ == "__main__":
    collect()