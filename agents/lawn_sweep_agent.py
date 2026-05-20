import numpy as np
from core.global_planner import AStarPlanner
from core.lawn_strip_follower import LawnStripFollower
from core.lawn_energy_manager import LawnEnergyManager
from core.config import ACTIONS, WAIT_ACTION
from collections import deque

class LawnSweepAgent:
    """
    Strip-following lawnmower agent с возвратом на базу.
    """

    def __init__(self):
        self.strip = LawnStripFollower()
        self.energy = LawnEnergyManager()
        self.planner = AStarPlanner()

        self.mode = "SWEEP"

        self.return_path = None
        self.replan_cooldown = 0
        self.replan_interval = 5

        self.prev_pos = None
        self.stuck_counter = 0
        self.recovery_path = None

    def reset(self):
        self.strip.reset()
        self.energy.reset()

        self.mode = "SWEEP"

        self.return_path = None
        self.replan_cooldown = 0

        self.prev_pos = None
        self.stuck_counter = 0
        self.recovery_path = None

    def act(self, env, temp=0):
        # =====================================================
        # 1. ЕСЛИ НА БАЗЕ — ЗАРЯДИТЬСЯ
        # =====================================================
        if env.pos == env.start_pos:
            self.energy.recharge_if_home(env)
            self.mode = "SWEEP"
            self.return_path = None
            self.recovery_path = None

        # =====================================================
        # 2. ENERGY HAS PRIORITY
        # =====================================================
        if env.pos != env.start_pos:
            if self.energy.should_return_home(env):
                self.mode = "RETURN_HOME"
                self.return_path = None
                self.recovery_path = None

        # =====================================================
        # 3. STUCK CHECK ТОЛЬКО ЕСЛИ НЕ ВОЗВРАЩАЕМСЯ
        # =====================================================
        if self.mode != "RETURN_HOME":
            if self.prev_pos == env.pos:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

            if self.stuck_counter >= 8:
                self.mode = "RECOVERY"
                self.recovery_path = None

        self.prev_pos = env.pos

        # =====================================================
        # 4. ACTION BY MODE
        # =====================================================
        if self.mode == "RETURN_HOME":
            action = self.act_return_home(env)

        elif self.mode == "RECOVERY":
            action = self.act_recovery(env)

        else:
            action = self.strip.act(env)

        debug = {
            "mode": self.mode,
            "strip_mode": self.strip.mode,
            "strip_direction": self.strip.direction,
            "lane_shift_dir": self.strip.lane_shift_dir,

            "return_path": self.return_path,
            "recovery_path": self.recovery_path,
            "path": self.return_path if self.mode == "RETURN_HOME" else self.recovery_path,

            "energy": env.energy_system.energy,
            "max_energy": env.energy_system.max_energy,
            "return_cost_est": self.energy.estimate_return_cost(env),

            "coverage_rate": getattr(env.env, "coverage_rate", lambda: 0.0)(),
            "overlap_rate": getattr(env.env, "overlap_rate", lambda: 0.0)(),

            "stuck_counter": self.stuck_counter,
            "goal": env.start_pos if self.mode == "RETURN_HOME" else None,
            "lane_memory": self.strip.memory.progress_report(env),
        }

        return action, debug

    def act_return_home(self, env):
        """
        Возврат домой через A*.
        """

        need_replan = False

        if self.return_path is None:
            need_replan = True
        elif len(self.return_path) < 2:
            need_replan = True
        elif self.replan_cooldown <= 0:
            need_replan = True
        elif self.path_is_blocked(env, self.return_path):
            need_replan = True

        if need_replan:
            self.return_path = self.find_cut_only_path_home(env)

            if self.return_path is not None:
                env.env.knife_on = False
            else:
                # безопасного скошенного коридора нет:
                # возвращаемся обычным A*, но нож должен быть включен
                env.env.knife_on = True

                self.return_path = self.planner.find_path_oriented(
                    env,
                    env.pos,
                    env.start_pos,
                    memory=None,
                    unknown_policy="allow",
                    robot_id="lawnmower",
                    blackboard=None,
                )

            self.replan_cooldown = self.replan_interval

        else:
            self.replan_cooldown -= 1

        if self.return_path is not None and len(self.return_path) >= 2:
            return self.action_from_path(env, self.return_path)

        return WAIT_ACTION

    def action_from_path(self, env, path):
        x, y = env.pos
        nx, ny = path[1]

        dx = nx - x
        dy = ny - y

        for i, (adx, ady) in enumerate(ACTIONS[:4]):
            if (adx, ady) == (dx, dy):
                return i

        return WAIT_ACTION

    def path_is_blocked(self, env, path):
        if path is None or len(path) < 2:
            return False

        return path[1] in getattr(env, "obstacles", set())

    def act_recovery(self, env):
        if self.recovery_path is not None and env.pos in self.recovery_path:
            idx = self.recovery_path.index(env.pos)
            self.recovery_path = self.recovery_path[idx:]

        if self.recovery_path is None or len(self.recovery_path) < 2:
            target = self.find_nearest_uncut(env)

            if target is None:
                self.mode = "SWEEP"
                return WAIT_ACTION

            self.recovery_path = self.planner.find_path_oriented(
                env,
                env.pos,
                target,
                memory=None,
                unknown_policy="allow",
                robot_id="lawnmower",
                blackboard=None,
            )

        if self.recovery_path is not None and len(self.recovery_path) >= 2:
            return self.action_from_path(env, self.recovery_path)

        self.mode = "SWEEP"
        return self.strip.act(env)

    def find_nearest_uncut(self, env):
        cells = np.argwhere(env.grid == 1)

        if len(cells) == 0:
            return None

        x, y = env.pos
        dists = np.abs(cells - np.array([x, y])).sum(axis=1)
        target = cells[np.argmin(dists)]

        return tuple(map(int, target))

    def find_cut_only_path_home(self, env):
        """
        Ищет путь домой только по уже скошенной зоне.
        adapter grid:
            1 = uncut grass
            2 = cut grass
            0 = empty
           -1 = obstacle/buffer
        """

        start = env.pos
        goal = env.start_pos

        q = deque([start])
        parent = {start: None}

        while q:
            p = q.popleft()

            if p == goal:
                break

            x, y = p

            for a, (dx, dy) in enumerate(ACTIONS[:4]):
                nx = x + dx
                ny = y + dy
                np_ = (nx, ny)

                if np_ in parent:
                    continue

                if nx < 0 or ny < 0:
                    continue

                if nx >= env.grid.shape[0] or ny >= env.grid.shape[1]:
                    continue

                if np_ in getattr(env, "obstacles", set()):
                    continue

                cell = env.grid[nx, ny]

                # можно ехать домой без ножа только по скошенному
                if cell != 2 and np_ != goal:
                    continue

                parent[np_] = p
                q.append(np_)

        if goal not in parent:
            return None

        path = []
        cur = goal

        while cur is not None:
            path.append(cur)
            cur = parent[cur]

        path.reverse()
        return path