from core.global_planner import AStarPlanner
from core.lawn_strip_follower import LawnStripFollower
from core.lawn_energy_manager import LawnEnergyManager
from core.config import ACTIONS, WAIT_ACTION


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

    def reset(self):
        self.strip.reset()
        self.energy.reset()

        self.mode = "SWEEP"

        self.return_path = None
        self.replan_cooldown = 0

    def act(self, env, temp=0):
        # =====================================================
        # 1. ЕСЛИ НА БАЗЕ — ЗАРЯДИТЬСЯ
        # =====================================================
        if env.pos == env.start_pos:
            self.energy.recharge_if_home(env)

            if self.mode == "RETURN_HOME":
                self.mode = "SWEEP"
                self.return_path = None

        # =====================================================
        # 2. РЕШЕНИЕ О ВОЗВРАТЕ
        # =====================================================
        if self.mode == "SWEEP":
            if self.energy.should_return_home(env):
                self.mode = "RETURN_HOME"
                self.return_path = None
                self.replan_cooldown = 0

        # =====================================================
        # 3. RETURN HOME MODE
        # =====================================================
        if self.mode == "RETURN_HOME":
            action = self.act_return_home(env)

        # =====================================================
        # 4. SWEEP MODE
        # =====================================================
        else:
            action = self.strip.act(env)

        debug = {
            "mode": self.mode,
            "strip_mode": self.strip.mode,
            "strip_direction": self.strip.direction,
            "lane_shift_dir": self.strip.lane_shift_dir,
            "return_path_len": 0 if self.return_path is None else len(self.return_path),
            "energy": env.energy_system.energy,
            "max_energy": env.energy_system.max_energy,
            "return_cost_est": self.energy.estimate_return_cost(env),
            "coverage_rate": getattr(env.env, "coverage_rate", lambda: 0.0)(),
            "overlap_rate": getattr(env.env, "overlap_rate", lambda: 0.0)(),
            "goal": env.start_pos if self.mode == "RETURN_HOME" else None,
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