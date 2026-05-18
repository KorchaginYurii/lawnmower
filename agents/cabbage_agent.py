import torch
import numpy as np
from collections import deque

from core.network import Net
from core.mcts import MCTS
from core.mcts import extract_rollout

from core.config import DEVICE, ACTIONS, VISION_SIZE

class CabbageAgent:
    def __init__(self):
        self.device = DEVICE

        self.net = Net().to(self.device)

        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-4)

        self.memory = deque(maxlen=50000)

        self.mcts = MCTS(self, simulations=10)
        #self.mcts = MCTS(
        #    self,
        #    simulations=10,
        #    batch_size=16
        #)

        print("🚀 DEVICE:", self.device)  # cuda или cpu

    def get_state(self, env):
        half = VISION_SIZE // 2
        h, w = env.grid.shape

        x, y = env.pos

        # ===== PAD ВСЕГО =====
        grid_pad = np.pad(env.grid, half)
        visited_pad = np.pad(env.visited, half)
        flood_pad = np.pad(env.flood_map, half)
        danger_pad = np.pad(env.danger_map, half)

        x_p = x + half
        y_p = y + half

        # ===== ОКНО =====
        window = grid_pad[x_p - half:x_p + half + 1, y_p - half:y_p + half + 1]
        visited_window = visited_pad[x_p - half:x_p + half + 1, y_p - half:y_p + half + 1]
        flood = flood_pad[x_p - half:x_p + half + 1, y_p - half:y_p + half + 1]
        danger = danger_pad[x_p - half:x_p + half + 1, y_p - half:y_p + half + 1]

        # ===== КАНАЛЫ =====
        cabbages = (window == 1).astype(np.float32)
        obstacles = (window == -1).astype(np.float32)

        # ===== блокировка старта ДО завершения =====
        rem = np.sum(env.grid == 1)

        if rem > 0:
            sx, sy = env.start_pos

            # координаты старта в окне
            sx_w = sx - (x - half)
            sy_w = sy - (y - half)

            if 0 <= sx_w < VISION_SIZE and 0 <= sy_w < VISION_SIZE:
                obstacles[sx_w, sy_w] = 1.0

        # ===== can_move (без цикла) =====
        can_move = np.zeros_like(window, dtype=np.float32)
        c = VISION_SIZE // 2

        # вручную 4 направления (быстрее любого цикла)
        if obstacles[c + 1, c] == 0: can_move[c + 1, c] = 1.0
        if obstacles[c - 1, c] == 0: can_move[c - 1, c] = 1.0
        if obstacles[c, c + 1] == 0: can_move[c, c + 1] = 1.0
        if obstacles[c, c - 1] == 0: can_move[c, c - 1] = 1.0

        # ===== dx / dy =====
        cabbages_global = np.argwhere(env.grid == 1)

        if len(cabbages_global) > 0:
            dists = np.abs(cabbages_global - np.array([x, y])).sum(axis=1)
            nearest = cabbages_global[np.argmin(dists)]

            dx = (nearest[0] - x) / h
            dy = (nearest[1] - y) / w
        else:
            dx, dy = 0.0, 0.0

        dx_map = np.full_like(window, dx, dtype=np.float32)
        dy_map = np.full_like(window, dy, dtype=np.float32)

        # ===== remaining =====
        remaining = np.sum(env.grid == 1) / 10.0
        remaining_map = np.full_like(window, remaining, dtype=np.float32)

        ones = np.ones_like(window, dtype=np.float32)

        # каналы стартовой позиции
        sx, sy = env.start_pos
        dx_home = np.full_like(window, (sx - (x - half)) / h)
        dy_home = np.full_like(window, (sy - (y - half)) / w)

        #🔥 ФАЗЫ

        remaining = np.sum(env.grid == 1)
        total = np.sum(env.initial_grid == 1)
        global_phase = 1.0 - (remaining / (total + 1e-6))
        global_phase_map = np.full_like(window, global_phase, dtype=np.float32)

        # ЛОКАЛЬНАЯ ФАЗА беру ближайшее → иду дальше искать
        window_area = VISION_SIZE * VISION_SIZE
        local_density = np.sum(window == 1) / window_area
        density_map = np.full_like(window, local_density, dtype=np.float32)
        local_phase = 1.0 - local_density
        local_phase_map = np.full_like(window, local_phase, dtype=np.float32)
        obstacle_density = np.full_like(window, env.obstacle_ratio, dtype=np.float32)
        # ===== STACK =====
        state = np.stack([
            cabbages,
            obstacles,
            visited_window,
            can_move,
            danger,
            flood,
            dx_map,
            dy_map,
            remaining_map,
            ones,
            dx_home,
            dy_home,
            global_phase_map,
            local_phase_map,
            density_map,
            obstacle_density
        ], axis=0)

        return state.astype(np.float32)

    def act(self, env, temp=1.0):
        probs, debug = self.mcts.run(env, temp=temp, training=(temp > 0))

        debug["rollout"] = extract_rollout(debug["root"])

        state = self.get_state(env)

        # ===== каналы =====
        c = state.shape[1] // 2  # центр окна

        debug["danger"] = state[4]
        debug["flood"] = state[5]

        # 🔥 ФАЗЫ (из state — правильно!)
        debug["local_phase"] = float(state[13, c, c])
        #debug["global_phase"] = float(state[12, c, c])
        remaining = np.sum(env.grid == 1)
        total = np.sum(env.initial_grid == 1)
        global_phase = 1.0 - remaining / (total + 1e-6)
        debug["global_phase"] = float(global_phase)

        # ===== distance до базы =====
        sx, sy = env.start_pos
        x, y = env.pos
        dist_home = abs(x - sx) + abs(y - sy)
        debug["dist_home"] = dist_home

        # ===== mode =====
        debug["mode"] = "RETURN" if debug["global_phase"] == 1.0 else "COLLECT"

        # ===== MCTS =====
        debug["mcts_sims"] = self.mcts.sims

        # ===== капуста =====
        total = np.sum(env.initial_grid == 1)
        remaining = np.sum(env.grid == 1)
        debug["cabbages_total"] = total
        debug["cabbages_collected"] = total - remaining

        # ===== шаги =====
        debug["step"] = env.steps
        debug["max_steps"] = env.max_steps


        # сохраняем debug
        self.last_debug = debug

        # ===== ACTION =====
        if temp == 0:
            return int(np.argmax(probs)), probs

        # повышение температуры в "залипших" местах
        if env.visited[env.pos] > 0.5:
            temp = 0.5

        probs = probs ** (1 / temp)
        probs /= probs.sum()

        return int(np.random.choice(4, p=probs)), probs

