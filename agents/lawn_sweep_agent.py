import numpy as np
from collections import deque

from core.config import ACTIONS, WAIT_ACTION
from core.global_planner import AStarPlanner
from core.lawn_strip_follower import LawnStripFollower
from core.lawn_energy_manager import LawnEnergyManager
from core.boustrophedon_decomposition import BoustrophedonDecomposition
from core.boustrophedon_mission_planner import BoustrophedonMissionPlanner
from core.cell_sweep_route import CellSweepRoute
from core.coverage_traffic_cost import CoverageTrafficCost

class LawnSweepAgent:
    """
    Lawn mower agent:
    - выбирает ближний сектор от базы
    - едет к сектору через A*
    - внутри сектора косит StripFollower'ом
    - при низкой энергии возвращается домой
    """

    def __init__(self):

        self.strip = LawnStripFollower()
        self.energy = LawnEnergyManager(reserve=30.0)

        self.planner = AStarPlanner()

        self.mode = "GO_TO_SECTOR"

        self.current_sector = None
        self.sector_target = None
        self.sector_path = None

        self.return_path = None
        self.recovery_path = None

        self.prev_pos = None
        self.stuck_counter = 0

        self.recent_positions = deque(maxlen=20)
        self.loop_counter = 0
        self.bad_action_counter = 0
        self.last_action = None

        self.last_cut = 0
        self.no_cut_counter = 0

        self.decomposition = BoustrophedonDecomposition()
        self.mission = BoustrophedonMissionPlanner()
        self.cell_route = CellSweepRoute()
        self.route_path = None

        self.decomposition_built = False
        self.current_cell = None

        self.traffic_cost = CoverageTrafficCost()

    def reset(self):
        self.strip.reset()
        self.energy.reset()


        self.mode = "GO_TO_SECTOR"

        self.current_sector = None
        self.sector_target = None
        self.sector_path = None

        self.return_path = None
        self.recovery_path = None

        self.prev_pos = None
        self.stuck_counter = 0

        self.recent_positions.clear()
        self.loop_counter = 0
        self.bad_action_counter = 0
        self.last_action = None

        self.last_cut = 0
        self.no_cut_counter = 0



        self.decomposition = BoustrophedonDecomposition()
        self.cell_route.reset()

        self.mission.reset()
        self.cell_route.reset()
        self.route_path = None

        self.decomposition_built = False
        self.current_cell = None

    def act(self, env, temp=0):
        # =====================================================
        # 0. FINISH HAS ABSOLUTE PRIORITY
        # =====================================================
        if env.env.remaining_grass() == 0:
            if env.pos == env.start_pos:
                self.mode = "FINISHED"
                return WAIT_ACTION, self.make_debug(env)

            self.mode = "RETURN_HOME"
            self.return_path = None
            self.sector_path = None
            self.recovery_path = None
            self.route_path = None

        # =====================================================
        # 1. HOME / RECHARGE / FINISH
        # =====================================================
        if env.pos == env.start_pos:
            self.energy.recharge_if_home(env)

            self.return_path = None
            self.recovery_path = None
            self.sector_path = None
            self.sector_target = None
            self.route_path = None

            self.stuck_counter = 0
            self.loop_counter = 0
            self.bad_action_counter = 0
            self.recent_positions.clear()

            self.strip.reset()

            if env.env.remaining_grass() == 0:
                self.mode = "FINISHED"
                return WAIT_ACTION, self.make_debug(env)

            # есть ещё трава — после зарядки снова выбираем coverage-cell
            self.mode = "GO_TO_SECTOR"
            self.current_cell = None
            self.mission.current_cell = None


        # =====================================================
        # 2. ENERGY PRIORITY
        # =====================================================
        if env.pos != env.start_pos and self.energy.should_return_home(env):
            self.mode = "RETURN_HOME"
            self.return_path = None
            self.sector_path = None
            self.recovery_path = None

        # =====================================================
        # 3. STUCK / LOOP CHECK
        # =====================================================
        if self.mode not in ("RETURN_HOME", "FINISHED") and env.pos != env.start_pos:
            if self.prev_pos == env.pos:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

            self.recent_positions.append(env.pos)

            unique_recent = len(set(self.recent_positions))

            if len(self.recent_positions) >= 20 and unique_recent <= 4:
                self.loop_counter += 1
            else:
                self.loop_counter = 0

            if self.stuck_counter >= 8 or self.loop_counter >= 3:
                self.mode = "RECOVERY"
                self.recovery_path = None
                self.strip.reset()

        self.prev_pos = env.pos

        # =====================================================
        # NO-CUT WATCHDOG
        # =====================================================
        current_cut = env.env.cut_grass()

        if self.mode == "SWEEP_SECTOR":
            if current_cut <= self.last_cut:
                self.no_cut_counter += 1
            else:
                self.no_cut_counter = 0

            if self.no_cut_counter >= 20:
                self.mode = "RECOVERY"
                self.recovery_path = None
                self.sector_path = None
                self.strip.reset()
                self.no_cut_counter = 0

        self.last_cut = current_cut

        # =====================================================
        # KNIFE CONTROL
        # =====================================================
        '''
        if self.mode in ("SWEEP_SECTOR", "GO_TO_SECTOR", "RECOVERY"):
            env.env.knife_on = True

        elif self.mode == "RETURN_HOME":
            # act_return_home сам может выключить нож,
            # если найден путь только по CUT
            pass

        elif self.mode == "FINISHED":
            env.env.knife_on = False
        '''

        # =====================================================
        # 4. ACTION BY MODE
        # =====================================================
        if self.mode == "RETURN_HOME":
            action = self.act_return_home(env)

        elif self.mode == "RECOVERY":
            action = self.act_recovery(env)

        elif self.mode == "GO_TO_SECTOR":
            action = self.act_go_to_sector(env)

        elif self.mode == "SWEEP_SECTOR":
            action = self.act_sweep_sector(env)

        elif self.mode == "FINISHED":
            action = WAIT_ACTION

        else:
            self.mode = "GO_TO_SECTOR"
            action = WAIT_ACTION

        # =====================================================
        # KNIFE CONTROL BY CELL
        # =====================================================
        if self.mode != "RETURN_HOME":
            env.env.knife_on = self.should_enable_knife(
                env,
                action,
            )

        # =====================================================
        # ACTION VALIDATION / SPIN PROTECTION
        # =====================================================
        if (
                self.mode not in ("FINISHED", "RETURN_HOME")
                and not self.is_action_valid(env, action)
        ):
            self.bad_action_counter += 1
        else:
            self.bad_action_counter = 0

        if self.bad_action_counter >= 3 and self.mode != "RETURN_HOME":
            self.mode = "GO_TO_SECTOR"
            self.recovery_path = None
            self.sector_path = None
            self.strip.reset()
            action = self.act_go_to_sector(env)
            self.bad_action_counter = 0

        self.last_action = action

        debug = self.make_debug(env)
        return action, debug

    # =====================================================
    # MODE: GO TO SECTOR
    # =====================================================

    def act_go_to_sector(self, env):
        self.ensure_decomposition(env)

        self.current_cell = self.mission.choose_cell(
            env,
            self.decomposition,
            self.traffic_cost,
        )

        if self.current_cell is None:
            self.mode = "RETURN_HOME"
            return self.act_return_home(env)

        current_cell = self.decomposition.cell_of(env.pos)

        if current_cell == self.current_cell:
            self.mode = "SWEEP_SECTOR"
            self.route_path = None
            return self.act_sweep_sector(env)

        target = self.decomposition.cells[self.current_cell].center()

        if target is None:
            self.mission.finish_cell(self.current_cell)
            self.current_cell = None
            self.mode = "GO_TO_SECTOR"
            return WAIT_ACTION

        need_replan = (
                self.sector_path is None
                or len(self.sector_path) < 2
                or self.sector_target != target
        )

        if need_replan:
            self.sector_target = target
            self.sector_path = self.planner.find_path_oriented(
                env,
                env.pos,
                target,
                memory=None,
                unknown_policy="allow",
                robot_id="lawnmower",
                blackboard=None,
            )

        if self.sector_path is not None:
            self.sector_path = self.sync_path(env, self.sector_path)

        if self.sector_path is not None and len(self.sector_path) >= 2:
            return self.action_from_path(env, self.sector_path)

        self.mode = "RECOVERY"
        self.recovery_path = None
        return WAIT_ACTION

    # =====================================================
    # MODE: SWEEP SECTOR
    # =====================================================

    def act_sweep_sector(self, env):
        self.ensure_decomposition(env)

        if self.current_cell is None:
            self.mode = "GO_TO_SECTOR"
            return WAIT_ACTION

        if self.mission.cell_uncut(
                env,
                self.decomposition,
                self.current_cell,
        ) <= 0:
            self.mission.finish_cell(self.current_cell)
            self.current_cell = None
            self.cell_route.reset()
            self.route_path = None
            self.mode = "GO_TO_SECTOR"
            return WAIT_ACTION

        current_cell = self.decomposition.cell_of(env.pos)

        if current_cell != self.current_cell:
            self.mode = "GO_TO_SECTOR"
            self.sector_path = None
            return self.act_go_to_sector(env)

        if self.cell_route.cell_id != self.current_cell:
            self.cell_route.build(
                env,
                self.decomposition,
                self.current_cell,
                start_pos=env.pos,
            )
            self.route_path = None

        self.cell_route.advance_if_reached(env)

        target = self.cell_route.current_waypoint(env)

        if target is None:
            self.mission.finish_cell(self.current_cell)
            self.current_cell = None
            self.cell_route.reset()
            self.route_path = None
            self.mode = "GO_TO_SECTOR"
            return WAIT_ACTION

        direct_action = self.direct_action_to_target(env, target)

        if direct_action is not None:
            self.route_path = None
            return direct_action

        if (
                self.route_path is None
                or len(self.route_path) < 2
                or target not in self.route_path
        ):
            self.route_path = self.planner.find_path_oriented(
                env,
                env.pos,
                target,
                memory=None,
                unknown_policy="allow",
                robot_id="lawnmower",
                blackboard=None,
            )

        if self.route_path is not None:
            self.route_path = self.sync_path(env, self.route_path)

        if self.route_path is not None and len(self.route_path) >= 2:
            return self.action_from_path(env, self.route_path)

        self.cell_route.route_index += 1
        self.route_path = None
        return WAIT_ACTION

    def action_stays_in_sector(self, env, action, sector_id):
        x, y = env.pos
        dx, dy = ACTIONS[action]

        nx = x + dx
        ny = y + dy

        x1, x2, y1, y2 = self.mission.get_sector_bounds(
            sector_id,
            env.grid.shape,
        )

        return x1 <= nx < x2 and y1 <= ny < y2

    def safe_action_inside_sector(self, env, sector_id):
        best = []

        for a, (dx, dy) in enumerate(ACTIONS[:4]):
            x, y = env.pos
            nx = x + dx
            ny = y + dy

            if not self.action_stays_in_sector(env, a, sector_id):
                continue

            if nx < 0 or ny < 0:
                continue

            if nx >= env.grid.shape[0] or ny >= env.grid.shape[1]:
                continue

            if (nx, ny) in getattr(env, "obstacles", set()):
                continue

            if env.grid[nx, ny] not in (1, 2):
                continue

            score = 0.0

            if env.grid[nx, ny] == 1:
                score -= 100.0
            else:
                score += 10.0

            if hasattr(env.env, "visit_count"):
                score += env.env.visit_count[nx, ny] * 5.0

            if (nx, ny) in self.recent_positions:
                score += 20.0

            best.append((score, a))

        if not best:
            self.mode = "RECOVERY"
            self.recovery_path = None
            return WAIT_ACTION

        best.sort(key=lambda z: z[0])

        best.sort(key=lambda z: z[0])

        best_score, best_action = best[0]

        if best_score > 20:
            self.mode = "RECOVERY"
            self.recovery_path = None
            return WAIT_ACTION

        return best_action

        return best[0][1]

    # =====================================================
    # MODE: RETURN HOME
    # =====================================================

    def act_return_home(self, env):
        # сначала ищем путь только по скошенному
        if self.return_path is None or len(self.return_path) < 2:
            self.return_path = self.find_cut_only_path_home(env)

            if self.return_path is not None:
                env.env.knife_on = False
            else:
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

        if self.return_path is not None:
            self.return_path = self.sync_path(env, self.return_path)

        if self.return_path is not None and len(self.return_path) >= 2:
            action = self.action_from_path(env, self.return_path)

            if self.return_path_is_cut_only(env, self.return_path):
                env.env.knife_on = False
            else:
                env.env.knife_on = self.should_enable_knife(env, action)

            return action

        return WAIT_ACTION

    def find_cut_only_path_home(self, env):
        start = env.pos
        goal = env.start_pos

        q = deque([start])
        parent = {start: None}

        while q:
            p = q.popleft()

            if p == goal:
                break

            x, y = p

            for dx, dy in ACTIONS[:4]:
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

    # =====================================================
    # MODE: RECOVERY
    # =====================================================

    def act_recovery(self, env):
        if env.pos == env.start_pos:
            self.mode = "GO_TO_SECTOR"
            self.recovery_path = None
            self.sector_path = None
            self.current_sector = None
            self.mission.current_sector = None
            self.strip.reset()
            return self.act_go_to_sector(env)

        if self.recovery_path is not None:
            self.recovery_path = self.sync_path(env, self.recovery_path)

        if self.recovery_path is None or len(self.recovery_path) < 2:

            target = self.find_best_recovery_target(env)

            if self.recovery_path is not None and len(self.recovery_path) >= 2:
                return self.action_from_path(env, self.recovery_path)

            if target is None:
                self.mode = "GO_TO_SECTOR"
                self.recovery_path = None
                return self.act_go_to_sector(env)

            self.recovery_path = self.planner.find_path_oriented(
                env,
                env.pos,
                target,
                memory=None,
                unknown_policy="allow",
                robot_id="lawnmower",
                blackboard=None,
            )

            if self.recovery_path is None or len(self.recovery_path) < 2:
                self.mode = "GO_TO_SECTOR"
                self.recovery_path = None
                return self.act_go_to_sector(env)

        action = self.action_from_path(env, self.recovery_path)

        if not self.is_action_valid(env, action):
            self.mode = "GO_TO_SECTOR"
            self.recovery_path = None
            self.sector_path = None
            self.strip.reset()
            return self.act_go_to_sector(env)

        return action

    def find_nearest_uncut_in_current_or_near_sector(self, env):
        if self.current_sector is not None:
            target = self.mission.nearest_uncut_in_sector(
                env,
                self.current_sector,
            )
            if target is not None:
                return target

        cells = np.argwhere(env.grid == 1)

        if len(cells) == 0:
            return None

        x, y = env.pos
        dists = np.abs(cells - np.array([x, y])).sum(axis=1)
        target = cells[np.argmin(dists)]

        return tuple(map(int, target))

    # =====================================================
    # HELPERS
    # =====================================================

    def sync_path(self, env, path):
        if path is None:
            return None

        if env.pos in path:
            idx = path.index(env.pos)
            return path[idx:]

        return None

    def action_from_path(self, env, path):
        if path is None or len(path) < 2:
            return WAIT_ACTION

        x, y = env.pos
        nx, ny = path[1]

        dx = nx - x
        dy = ny - y

        for i, (adx, ady) in enumerate(ACTIONS[:4]):
            if (adx, ady) == (dx, dy):
                return i

        return WAIT_ACTION

    def make_debug(self, env):
        mission_debug = self.mission.debug()
        decomp_debug = (
            self.decomposition.debug_info()
            if self.decomposition_built
            else {"coverage_cells": 0}
        )
        return {
            "mode": self.mode,

            "sector": mission_debug.get("current_cell"),
            "sector_h": 10,
            "sector_w": 10,
            "sector_uncut": (
                0
                if self.current_cell is None or not self.decomposition_built
                else self.mission.cell_uncut(env, self.decomposition, self.current_cell)
            ),
            "sector_coverage": (
                1.0
                if self.current_cell is None or not self.decomposition_built
                else self.mission.cell_coverage(env, self.decomposition, self.current_cell)
            ),

            "strip_mode": self.strip.mode,
            "strip_direction": self.strip.direction,
            "lane_shift_dir": self.strip.lane_shift_dir,

            "path": (
                self.return_path
                if self.mode == "RETURN_HOME"
                else self.sector_path
                if self.mode == "GO_TO_SECTOR"
                else self.recovery_path
            ),

            "return_path": self.return_path,
            "recovery_path": self.recovery_path,

            "energy": env.energy_system.energy,
            "max_energy": env.energy_system.max_energy,
            "return_cost_est": self.energy.estimate_return_cost(env),

            "coverage_rate": getattr(env.env, "coverage_rate", lambda: 0.0)(),
            "overlap_rate": getattr(env.env, "overlap_rate", lambda: 0.0)(),

            "stuck_counter": self.stuck_counter,
            "goal": (
                env.start_pos
                if self.mode == "RETURN_HOME"
                else self.sector_target
            ),

            "lane_memory": self.strip.memory.progress_report(env),
            "knife_on": getattr(env.env, "knife_on", True),
            "loop_counter": self.loop_counter,
            "no_cut_counter": self.no_cut_counter,

            "current_cell": mission_debug.get("current_cell"),
            "last_cell": mission_debug.get("last_cell"),
            "coverage_cells": decomp_debug.get("coverage_cells", 0),

            **self.cell_route.debug(),

            #"knife_on_real": self.knife_on,
        }

    def is_action_valid(self, env, action):
        if action == WAIT_ACTION:
            return False

        x, y = env.pos
        dx, dy = ACTIONS[action]

        nx = x + dx
        ny = y + dy

        if nx < 0 or ny < 0:
            return False

        if nx >= env.grid.shape[0] or ny >= env.grid.shape[1]:
            return False

        if (nx, ny) in getattr(env, "obstacles", set()):
            return False

        return env.grid[nx, ny] in (1, 2)

    def direct_action_to_target(self, env, target):
        x, y = env.pos
        tx, ty = target

        dx = tx - x
        dy = ty - y

        if abs(dx) + abs(dy) != 1:
            return None

        for a, (adx, ady) in enumerate(ACTIONS[:4]):
            if (adx, ady) == (dx, dy):
                return a

        return None

    def find_best_recovery_target(self, env):
        """
        Выбирает достижимую нескошенную клетку.
        Не просто ближайшую по Manhattan, а ту, до которой реально есть путь.
        """

        cells = np.argwhere(env.grid == 1)

        if len(cells) == 0:
            return None

        x, y = env.pos

        candidates = []

        for tx, ty in cells:
            tx = int(tx)
            ty = int(ty)

            dist = abs(tx - x) + abs(ty - y)

            if dist > 30:
                continue

            visit_penalty = 0.0

            if hasattr(env.env, "visit_count"):
                visit_penalty = env.env.visit_count[tx, ty] * 5.0

            sector_penalty = 0.0

            if self.current_cell is not None and self.decomposition_built:
                if self.decomposition.cell_of((tx, ty)) != self.current_cell:
                    sector_penalty = 25.0

            score = dist + visit_penalty + sector_penalty

            candidates.append((score, (tx, ty)))

        candidates.sort(key=lambda z: z[0])

        targets = [
            target
            for _, target in candidates[:50]
        ]

        target, path = self.traffic_cost.best_path(
            env,
            self.planner,
            targets,
            max_targets=30,
        )

        if target is not None:
            self.recovery_path = path
            return target

        return None

    def ensure_decomposition(self, env):
        if not self.decomposition_built:
            self.decomposition.build(env)
            self.decomposition_built = True

    def should_enable_knife(self, env, action):
        """
        Нож включается только если сейчас или следующим шагом
        робот реально будет косить нескошенную траву.
        """
        if self.mode == "FINISHED":
            return False

        if action == WAIT_ACTION:
            return False

        x, y = env.pos
        dx, dy = ACTIONS[action]

        nx = x + dx
        ny = y + dy

        if not (0 <= nx < env.grid.shape[0] and 0 <= ny < env.grid.shape[1]):
            nx, ny = x, y

        return self.footprint_has_grass(env, (nx, ny))

    def footprint_has_grass(self, env, pos):
        """
        Проверяет не центр робота, а всю зону покоса 2x2.
        Должно совпадать с LawnEnv.cut_under_robot().
        """
        import math

        radius_cells = max(
            1,
            int(math.ceil((env.env.robot_size_m / 2.0) / env.env.cell_size_m))
        )

        cx, cy = pos

        for dx in range(-radius_cells + 1, radius_cells + 1):
            for dy in range(-radius_cells + 1, radius_cells + 1):
                x = cx + dx
                y = cy + dy

                if 0 <= x < env.grid.shape[0] and 0 <= y < env.grid.shape[1]:
                    if env.grid[x, y] == 1:
                        return True

        return False

    def return_path_is_cut_only(self, env, path):
        if path is None:
            return False

        for p in path[1:]:
            x, y = p

            if p == env.start_pos:
                continue

            if env.grid[x, y] != 2:
                return False

        return True