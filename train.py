import os
os.system("")

import re
import shutil
import random
import numpy as np
import torch

from collections import deque

from agents.cabbage_agent import CabbageAgent
from agents.hybrid_agent import HybridAgent
from env.cabbage_env import CabbageEnv

from core.config import EPISODES, BATCH_SIZE, GAMMA, MAP_H, MAP_W
from core.checkpoint import CheckpointManager
from core.metrics_logger import MetricsLogger


# =====================================================
# TORCH SETTINGS
# =====================================================
torch.backends.cudnn.benchmark = True


# =====================================================
# GLOBALS
# =====================================================
start_ep = 0
best = -1e9
best_r = -1e9
eval_score = -1

elite_memory = deque(maxlen=10000)

#MODE = "resume"
MODE = "pretrain"
# MODE = "scratch"


# =====================================================
# EVALUATE
# =====================================================
def evaluate(agent, n=10):
    scores = []
    success = 0


    for ev in range(n):
        env = CabbageEnv(MAP_H, MAP_W)
        env.reset()
        h, w = env.grid.shape

        total_reward = 0.0
        episode_limit = env.max_steps + max(h,w) * 2

        for st in range(episode_limit):
            a, _ = agent.act(env, temp=0)
            r, done = env.step(a)
            total_reward += r

            remaining = np.sum(env.grid == 1)
            total_cabbages = np.sum(env.initial_grid == 1)
            phase = 1.0 - remaining / (total_cabbages + 1e-6)

            ev_bar = int(10 * (ev + 1) / n)
            st_bar = int(20 * (st + 1) / episode_limit)

            avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0.0

            print(
                f"\r🚀 Eval [{'#' * ev_bar}{'-' * (10 - ev_bar)}] {ev + 1}/{n} | "
                f"STEP [{'#' * st_bar}{'-' * (20 - st_bar)}] {st + 1}/{episode_limit} | "
                f"phase {phase:.2f} | "
                f"score: {avg_score:.2f} | "
                f"success_rate: {(success / max(1, ev + 1)):.2f}",
                end="",
                flush=True
            )

            if done:
                break

        num_cabbages = max(1, np.sum(env.initial_grid == 1))
        norm_score = total_reward / num_cabbages

        all_collected = (np.sum(env.grid == 1) == 0)
        at_start = (env.pos == env.start_pos)

        if all_collected and at_start:
            success += 1

        scores.append(norm_score)

    print()

    return {
        "score": sum(scores) / len(scores),
        "success_rate": success / n
    }

# =====================================================
# PARTIAL LOAD
# =====================================================
def load_pretrained_partial(agent, path):
    print(f"🔄 Loading pretrained partial: {path}")

    checkpoint = torch.load(
        path,
        map_location=agent.device,
        weights_only=False
    )

    old_state = checkpoint["model"]
    new_state = agent.net.state_dict()

    with torch.no_grad():
        if "conv.0.weight" in old_state:
            old_w = old_state["conv.0.weight"]
            new_w = agent.net.conv[0].weight

            c_old = min(old_w.shape[1], new_w.shape[1])

            new_w[:, :c_old] = old_w[:, :c_old]

            if new_w.shape[1] > c_old:
                new_w[:, c_old:] = 0.0

            print("✅ Initialized compatible conv channels")

    filtered = {
        k: v for k, v in old_state.items()
        if k in new_state and v.shape == new_state[k].shape
    }

    new_state.update(filtered)
    agent.net.load_state_dict(new_state)

    print(f"✅ Loaded {len(filtered)} layers partial")

def find_best_model1(folder="checkpoints"):
    best_score = -1e9
    best_path = None

    if not os.path.exists(folder):
        return None

    for f in os.listdir(folder):
        if f.startswith("best_") and f.endswith(".pth"):
            match = re.search(r"score(-?\d+\.\d+)", f)

            if match:
                score = float(match.group(1))
                print(f, "->", score)

                if score > best_score:
                    best_score = score
                    best_path = os.path.join(folder, f)

    return best_path

# =====================================================
# INIT
# =====================================================
ckpt = CheckpointManager(
    k_best=3,
    project_name="Cab2"
)

agent = CabbageAgent()

logger = MetricsLogger()

if MODE == "resume":
    start_ep, best = ckpt.load_checkpoint(agent)

elif MODE == "pretrain":
    path = find_best_model1()

    if path is not None:
        load_pretrained_partial(agent, path)
    else:
        print("⚠️ No pretrained model found, starting fresh")

    start_ep = 0
    best = -1e9

    agent.memory.clear()
    elite_memory.clear()

else:
    start_ep = 0
    best = -1e9


# =====================================================
# MAIN LOOP
# =====================================================

for ep in range(start_ep, EPISODES):

    # =================================================
    # ENV
    # =================================================
    env = CabbageEnv(MAP_H, MAP_W)
    h = env.height
    w = env.width

    if ep < 100:
        env.reset(obs_min=0.00, obs_max=0.05, cab_min=0.25, cab_max=0.35)
    elif ep < 200:
        env.reset(obs_min=0.03, obs_max=0.10, cab_min=0.30, cab_max=0.45)
    elif ep < 500:
        env.reset(obs_min=0.08, obs_max=0.18, cab_min=0.40, cab_max=0.55)
    elif ep < 800:
        env.reset(obs_min=0.15, obs_max=0.25, cab_min=0.50, cab_max=0.65)
    else:
        env.reset(obs_min=0.20, obs_max=0.30, cab_min=0.55, cab_max=0.70)

    rewards = []
    states = []
    policies = []

    total_reward = 0.0
    done = False
    debug = {}


    episode_limit = env.max_steps + max(h,w) * 2

    # =================================================
    # MCTS SCHEDULE
    # =================================================
    if ep < 200:
        sims = 50
    elif ep < 500:
        sims = 100
    else:
        sims = 150

    density = np.sum(env.grid == 1) / (h * w)
    scale = 1 + (1 - density)

    if density < 0.2:
        sims *= 2
    elif density > 0.5:
        sims *= 0.7

    agent.mcts.sims = int(sims * scale)
    base_sims = agent.mcts.sims

    # =================================================
    # LR
    # =================================================
    if ep < 200:
        lr = 5e-5
    else:
        lr = 1e-4

    for g in agent.opt.param_groups:
        g["lr"] = lr

    # =================================================
    # TEMPERATURE
    # =================================================
    temp = max(0.1, 1.0 - ep / 300)

    # =================================================
    # EPISODE
    # =================================================
    for t in range(episode_limit):

        remaining = np.sum(env.grid == 1)
        total_cabbages = np.sum(env.initial_grid == 1)

        global_phase = 1.0 - remaining / (total_cabbages + 1e-6)

        if global_phase >= 0.999:
            agent.mcts.sims = int(base_sims * 2)
        elif global_phase > 0.8:
            agent.mcts.sims = int(base_sims * 1.3)
        else:
            agent.mcts.sims = base_sims

        ep_bar = int(15 * ep / EPISODES)
        t_bar = int(15 * (t + 1) / episode_limit)

        local_phase = 0.0
        if hasattr(agent, "last_debug"):
            local_phase = agent.last_debug.get("local_phase", 0.0)

        at_start = (env.pos == env.start_pos)

        print(
            f"\rEP [{'#' * ep_bar}{'-' * (15 - ep_bar)}] {ep}/{EPISODES} | "
            f"STEP [{'#' * t_bar}{'-' * (15 - t_bar)}] {t + 1}/{episode_limit} | "
            f"R {total_reward:.1f} | "
            f"local {local_phase:.2f} | "
            f"global {global_phase:.2f} | "
            f"home {at_start}",
            end="",
            flush=True
        )

        # ===== ACT =====
        a, pi = agent.act(env, temp=temp)
        debug = getattr(agent, "last_debug", {})

        # ===== STOCHASTICITY =====
        if ep < 100:
            noise_prob = 0.02
        else:
            noise_prob = max(0.05, 0.2 - ep / 1000)

        if random.random() < noise_prob:
            a = random.randint(0, 3)

        # ===== STORE STATE BEFORE STEP =====
        s = agent.get_state(env)

        states.append(s)
        policies.append(pi)

        # ===== STEP =====
        r, done = env.step(a)

        rewards.append(r)
        total_reward += r

        if done:
            break

    # =================================================
    # RETURNS
    # =================================================
    if not done:
        last_state = agent.get_state(env)

        with torch.no_grad():
            _, v = agent.net(
                torch.tensor(
                    last_state,
                    dtype=torch.float32,
                    device=agent.device
                ).unsqueeze(0)
            )

            G = v.item()
    else:
        G = 0.0

    returns = []

    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.append(G)

    returns.reverse()
    returns = np.array(returns, dtype=np.float32)

    returns = returns / 100.0

    # =================================================
    # MEMORY
    # =================================================
    for s, pi, v in zip(states, policies, returns):
        agent.memory.append((s, pi, v))

    # =================================================
    # EPISODE METRICS
    # =================================================
    num_cabbages = np.sum(env.initial_grid == 1)
    collected = num_cabbages - np.sum(env.grid == 1)

    max_possible = num_cabbages * 10 + 100
    score = total_reward / (max_possible + 1e-6)

    all_collected = (np.sum(env.grid == 1) == 0)
    at_start = (env.pos == env.start_pos)

    success = int(all_collected and at_start)

    if success:
        fail_reason = "success"
    elif all_collected and not at_start:
        fail_reason = "no_return"
    else:
        fail_reason = "not_collected"

    # =================================================
    # ELITE MEMORY
    # =================================================
    norm_score = total_reward / (num_cabbages + 1)

    if norm_score > 8:
        for s, pi, v in zip(states, policies, returns):
            if abs(v) > 0.3:
                elite_memory.append((s, pi, v))

    # =================================================
    # TRAIN STEP
    # =================================================
    loss_value = 0.0

    if len(agent.memory) >= BATCH_SIZE:

        main_size = int(BATCH_SIZE * 0.7)
        elite_size = BATCH_SIZE - main_size

        batch_main = random.sample(agent.memory, main_size)

        batch_elite = []
        if len(elite_memory) > 0:
            batch_elite = random.sample(
                elite_memory,
                min(len(elite_memory), elite_size)
            )

        batch = batch_main + batch_elite

        states_b, pi_b, v_b = zip(*batch)

        states_b = torch.from_numpy(
            np.stack(states_b)
        ).float().to(agent.device)

        pi_b = torch.from_numpy(
            np.array(pi_b)
        ).float().to(agent.device)

        v_b = torch.from_numpy(
            np.array(v_b)
        ).float().to(agent.device)

        logits, v_pred = agent.net(states_b)
        v_pred = v_pred.squeeze()

        value_loss = ((v_pred - v_b) ** 2).mean()

        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        policy_loss = -(pi_b * log_probs).sum(dim=1).mean()
        entropy = -(probs * log_probs).sum(dim=1).mean()

        loss = 0.5 * value_loss + policy_loss - 0.05 * entropy

        agent.opt.zero_grad()
        loss.backward()
        agent.opt.step()

        loss_value = loss.item()

    # =================================================
    # SAVE LAST
    # =================================================
    ckpt.save_last(agent, ep, best)

    # =================================================
    # LOG CSV
    # =================================================
    logger.log(
        episode=ep,
        success=int(success),
        fail_reason=fail_reason,
        reward=total_reward,
        collected=collected,
        total_cabbages=num_cabbages,
        steps=env.steps,
        energy_used=getattr(env, "energy_used", 0.0),
        energy_per_cabbage=debug.get("energy_per_cabbage", 0),
        total_turns=getattr(env, "total_turns", 0),
        overlap_rate=debug.get("overlap_rate", 0),
        sector_switches=debug.get("sector_switches", 0),
    )

    # =================================================
    # PRINT
    # =================================================
    if ep % 10 == 0:
        print(
            f"\nEp {ep} | "
            f"R={total_reward:.1f} | "
            f"norm={score:.2f} | "
            f"cabbages={collected}/{num_cabbages} | "
            f"home={at_start} | "
            f"limit={episode_limit} | "
            f"loss={loss_value:.4f} | "
            f"success={fail_reason}"
        )

    # =================================================
    # EVAL + SAVE BEST
    # =================================================
    if ep % 20 == 0:
        eval_result = evaluate(agent, n=5)

        eval_score = eval_result["score"]
        eval_success = eval_result["success_rate"]

        if eval_success > best:
            best = eval_success

            ckpt.save(agent, ep, best, eval_score)

            ckpt_dir = "/kaggle/working/Cab2/checkpoints"
            zip_path = "/kaggle/working/checkpoints_backup"

            if os.path.exists(ckpt_dir):
                shutil.make_archive(zip_path, "zip", ckpt_dir)
                print(f"\n✅ Checkpoints archived: {zip_path}.zip")