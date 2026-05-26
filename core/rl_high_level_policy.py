import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RLPolicyNet(nn.Module):
    def __init__(self, state_dim=7, action_dim=5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class RLHighLevelPolicy:
    """
    RL controller поверх planner.

    Actions:
        0 = CONSERVATIVE
        1 = BALANCED
        2 = AGGRESSIVE
        3 = RETURN_HOME
        4 = RECOVERY
    """

    def __init__(
        self,
        state_dim=7,
        action_dim=5,
        lr=1e-3,
        gamma=0.99,
        epsilon=0.15,
        model_path=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon

        self.policy = RLPolicyNet(
            state_dim,
            action_dim,
        )

        if model_path is not None:
            self.load(model_path)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=lr,
        )

        self.last_state = None
        self.last_action = 1

    def reset(self):
        self.last_state = None
        self.last_action = 1

    def load(self, path, device="cpu"):
        self.policy.load_state_dict(
            torch.load(
                path,
                map_location=device,
            )
        )

        self.policy.eval()

        print(f"✅ RL policy loaded: {path}")

    def state(self, env, agent):
        coverage = env.env.coverage_rate()
        overlap = env.env.overlap_rate()

        energy = env.energy_system.energy
        max_energy = env.energy_system.max_energy
        energy_ratio = energy / max(1e-9, max_energy)

        remaining = env.env.remaining_grass()
        total = env.env.total_grass()
        remaining_ratio = remaining / max(1, total)

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

    def act(self, env, agent, train=True):
        s = self.state(env, agent)

        if train and random.random() < self.epsilon:
            a = random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                x = torch.tensor(
                    s,
                    dtype=torch.float32,
                ).unsqueeze(0)

                logits = self.policy(x)

                a = int(torch.argmax(logits, dim=1).item())

        self.last_state = s
        self.last_action = a

        return a

    def apply(self, action, runtime_config):
        if action == 0:  # CONSERVATIVE
            runtime_config.set("VISIT_WEIGHT", 0.07)
            runtime_config.set("CUT_WEIGHT", 0.10)
            runtime_config.set("CELL_TRAFFIC_WEIGHT", 1.1)

        elif action == 1:  # BALANCED
            runtime_config.set("VISIT_WEIGHT", 0.04)
            runtime_config.set("CUT_WEIGHT", 0.08)
            runtime_config.set("CELL_TRAFFIC_WEIGHT", 0.8)

        elif action == 2:  # AGGRESSIVE
            runtime_config.set("VISIT_WEIGHT", 0.03)
            runtime_config.set("CUT_WEIGHT", 0.06)
            runtime_config.set("CELL_TRAFFIC_WEIGHT", 0.6)

        elif action == 3:  # RETURN_HOME
            runtime_config.set("ENERGY_RESERVE", 35.0)

        elif action == 4:  # RECOVERY
            runtime_config.set("VISIT_WEIGHT", 0.08)
            runtime_config.set("CELL_TRAFFIC_WEIGHT", 1.2)

    def debug(self):
        names = {
            0: "RL_CONSERVATIVE",
            1: "RL_BALANCED",
            2: "RL_AGGRESSIVE",
            3: "RL_RETURN_HOME",
            4: "RL_RECOVERY",
        }

        return {
            "rl_action": self.last_action,
            "rl_mode": names.get(self.last_action, "UNKNOWN"),
        }

