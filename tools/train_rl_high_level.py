import os
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.lawn_env import LawnEnv
from adapters.lawn_hybrid_adapter import LawnHybridAdapter
from agents.lawn_sweep_agent import LawnSweepAgent

from core.rl_high_level_policy import RLPolicyNet
from core.tuning_config import runtime_config
from core.config import (
    LAWN_PRESET,
    LAWN_PRESETS,
    CELL_SIZE_M,
    ROBOT_SIZE_M,
    LAWNMOWER_MAX_ENERGY,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES = 300
MAX_STEPS_MULT = 3

GAMMA = 0.98
LR = 3e-4

BATCH_SIZE = 64
MEMORY_SIZE = 20_000

EPS_START = 0.08
EPS_END = 0.01
EPS_DECAY = 250

TARGET_UPDATE = 20

SAVE_DIR = "checkpoints/rl_high_level"


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        s, a, r, ns, done = zip(*batch)

        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=DEVICE),
            torch.tensor(a, dtype=torch.long, device=DEVICE),
            torch.tensor(r, dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(ns), dtype=torch.float32, device=DEVICE),
            torch.tensor(done, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


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


def hl_state(env, agent):
    coverage = env.env.coverage_rate()
    overlap = env.env.overlap_rate()

    energy_ratio = (
        env.energy_system.energy
        / max(1e-9, env.energy_system.max_energy)
    )

    remaining_ratio = (
        env.env.remaining_grass()
        / max(1, env.env.total_grass())
    )

    return np.array(
        [
            coverage,
            overlap,
            energy_ratio,
            remaining_ratio,
            agent.no_cut_counter / 50.0,
            agent.loop_counter / 10.0,
            agent.stuck_counter / 10.0,
        ],
        dtype=np.float32,
    )


def apply_hl_action(action):
    if action == 0:      # conservative
        runtime_config.set("VISIT_WEIGHT", 0.07)
        runtime_config.set("CUT_WEIGHT", 0.10)
        runtime_config.set("CELL_TRAFFIC_WEIGHT", 1.1)

    elif action == 1:    # balanced
        runtime_config.set("VISIT_WEIGHT", 0.04)
        runtime_config.set("CUT_WEIGHT", 0.08)
        runtime_config.set("CELL_TRAFFIC_WEIGHT", 0.8)

    elif action == 2:    # aggressive
        runtime_config.set("VISIT_WEIGHT", 0.03)
        runtime_config.set("CUT_WEIGHT", 0.06)
        runtime_config.set("CELL_TRAFFIC_WEIGHT", 0.6)

    elif action == 3:    # return home bias
        runtime_config.set("ENERGY_RESERVE", 35.0)

    elif action == 4:    # recovery bias
        runtime_config.set("VISIT_WEIGHT", 0.08)
        runtime_config.set("CELL_TRAFFIC_WEIGHT", 1.2)


def hl_reward(lawn, prev_coverage, prev_overlap):
    coverage = lawn.coverage_rate()
    overlap = lawn.overlap_rate()

    d_cov = coverage - prev_coverage
    d_overlap = max(0.0, overlap - prev_overlap)

    reward = 0.0

    reward += d_cov * 300.0
    reward -= d_overlap * 80.0

    reward -= 0.02
    reward -= lawn.energy_used * 0.0001

    if lawn.remaining_grass() == 0 and lawn.pos == lawn.start_pos:
        reward += 50.0

    if lawn.energy <= 0:
        reward -= 30.0

    return reward


def choose_action(policy, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(5)

    with torch.no_grad():
        x = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        q = policy(x)

        return int(torch.argmax(q, dim=1).item())


def optimize(policy, target, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return None

    s, a, r, ns, done = memory.sample(BATCH_SIZE)

    q = policy(s).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        nq = target(ns).max(1)[0]
        tq = r + GAMMA * nq * (1.0 - done)

    loss = nn.functional.smooth_l1_loss(q, tq)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def save_model(policy, episode, score):
    os.makedirs(SAVE_DIR, exist_ok=True)

    path = os.path.join(
        SAVE_DIR,
        f"rl_high_level_ep{episode}_score{score:.1f}.pth",
    )

    torch.save(policy.state_dict(), path)

    print(f"✅ saved: {path}")


def train():
    runtime_config.reset()
    runtime_config.load_profile("configs.tuned_stable")
    runtime_config.set("USE_HIGH_LEVEL_POLICY", False)
    runtime_config.set("USE_RL_HIGH_LEVEL_POLICY", False)
    runtime_config.set("USE_ADAPTIVE_TRAFFIC", False)

    policy = RLPolicyNet().to(DEVICE)

    policy.load_state_dict(
        torch.load(
            "checkpoints/rl_high_level/"
            "rl_high_level_imitation_best.pth",
            map_location=DEVICE,
        )
    )
    target = RLPolicyNet().to(DEVICE)
    target.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    best_score = -1e18

    seeds = list(range(1000, 1000 + EPISODES))

    for ep in range(EPISODES):
        epsilon = max(
            EPS_END,
            EPS_START - ep / EPS_DECAY * (EPS_START - EPS_END),
        )

        seed = seeds[ep]

        lawn, env = make_env(seed)

        agent = LawnSweepAgent()
        agent.reset()

        max_steps = max(
            500,
            int(lawn.total_grass() * MAX_STEPS_MULT),
        )

        total_reward = 0.0
        losses = []

        state = hl_state(env, agent)

        prev_coverage = lawn.coverage_rate()
        prev_overlap = lawn.overlap_rate()

        for step in range(max_steps):
            env.sync_from_env()

            action_hl = choose_action(policy, state, epsilon)
            apply_hl_action(action_hl)

            action, debug = agent.act(env)

            reward_env, done = env.step(action)

            reward = hl_reward(
                lawn,
                prev_coverage,
                prev_overlap,
            )

            prev_coverage = lawn.coverage_rate()
            prev_overlap = lawn.overlap_rate()

            env.sync_from_env()

            next_state = hl_state(env, agent)

            terminal = bool(done)

            memory.push(
                state,
                action_hl,
                reward,
                next_state,
                terminal,
            )

            loss = optimize(
                policy,
                target,
                optimizer,
                memory,
            )

            if loss is not None:
                losses.append(loss)

            total_reward += reward
            state = next_state

            if done:
                break

        if ep % TARGET_UPDATE == 0:
            target.load_state_dict(policy.state_dict())

        coverage = lawn.coverage_rate()
        overlap = lawn.overlap_rate()
        success = int(
            lawn.remaining_grass() == 0
            and lawn.pos == lawn.start_pos
        )

        score = (
            success * 3000
            + coverage * 700
            - overlap * 500
            - (lawn.energy_used / max(1, lawn.cut_grass())) * 300
            - lawn.steps * 0.04
            - lawn.total_turns * 0.15
        )

        avg_loss = np.mean(losses) if losses else 0.0

        print(
            f"EP {ep:04d} | "
            f"seed={seed} | "
            f"success={success} | "
            f"cov={coverage:.3f} | "
            f"ov={overlap:.3f} | "
            f"steps={lawn.steps} | "
            f"R={total_reward:.2f} | "
            f"score={score:.1f} | "
            f"eps={epsilon:.3f} | "
            f"loss={avg_loss:.4f}"
        )

        if score > best_score:
            best_score = score
            save_model(policy, ep, score)

    final_path = os.path.join(
        SAVE_DIR,
        f"rl_high_level_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
    )

    torch.save(policy.state_dict(), final_path)

    print(f"\n✅ final saved: {final_path}")


if __name__ == "__main__":
    train()