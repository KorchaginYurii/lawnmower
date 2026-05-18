import numpy as np


from core.config import ACTIONS, DIRECTIONS, MOVE_COST, TURN_COST, CUT_COST
from core.sector_manager import SectorManager
from core.sector_coverage import SectorCoveragePlanner
from core.energy_predictor import EnergyPredictor
from core.world_memory import WorldMemory
from core.frontier_manager import FrontierManager
from core.team_blackboard import TeamBlackboard
from core.tuning_config import runtime_config
from core.mission_planner import MissionPlanner
from core.config import (
    OPPORTUNISTIC_RETURN_MARGIN,
    OPPORTUNISTIC_MAX_EXTRA_COST,
    OPPORTUNISTIC_MIN_CABBAGES,
)
from core.failure_recovery import FailureRecoveryManager
from core.config import WAIT_ACTION, USE_HIERARCHICAL_PLANNER
from core.config import USE_PORTAL_PLANNER
from core.global_planner import AStarPlanner
from core.portal_planner import PortalPlanner
from core.hierarchical_planner import HierarchicalPlanner

class HybridAgent:
    def __init__(self, local_agent=None, robot_id="robot_1", blackboard=None):
        self.local_agent = local_agent
        self.robot_id = robot_id
        self.blackboard = blackboard or TeamBlackboard()

        if self.local_agent is None:
            print("⚠️ HybridAgent without local RL agent")
        else:
            print("✅ HybridAgent using RL local_agent")
        self.sectors = SectorManager()

        self.low_planner = AStarPlanner()

        if USE_PORTAL_PLANNER:
            self.planner = PortalPlanner(self.sectors, self.low_planner)

        elif USE_HIERARCHICAL_PLANNER:
            self.planner = HierarchicalPlanner(self.sectors, self.low_planner)

        else:
            self.planner = self.low_planner

        self.coverage = SectorCoveragePlanner()
        self.mode = "COLLECT"
        self.goal = None
        self.path = []
        self.energy_predictor = EnergyPredictor(reserve=5.0)
        self.last_sector = None
        self.sector_switches = 0
        self.memory = WorldMemory()
        self.frontiers = FrontierManager()
        self.replan_interval = runtime_config.get("REPLAN_INTERVAL", 8)
        self.replan_cooldown = 0
        self.prev_pos = None
        self.mission = MissionPlanner()
        self.recovery = FailureRecoveryManager()

    def reset(self):

        self.mode = "COLLECT"
        self.goal = None
        self.path = []
        self.coverage.reset()
        self.last_sector = None
        self.sector_switches = 0
        self.memory = WorldMemory()
        self.replan_interval = 8
        self.replan_cooldown = 0
        self.prev_pos = None
        self.recovery.reset()

    def nearest_cabbage(self, env):
        cabbages = np.argwhere(env.grid == 1)

        if len(cabbages) == 0:
            return None

        x, y = env.pos

        dists = np.abs(cabbages - np.array([x, y])).sum(axis=1)
        nearest = cabbages[np.argmin(dists)]

        return tuple(nearest)

    def choose_goal(self, env):
        remaining = np.sum(env.grid == 1)

        # =====================================================
        # 1. ВСЁ СОБРАНО → ФИНИШ НА БАЗУ
        # =====================================================
        if remaining == 0:
            self.mode = "RETURN_FINISH"
            return env.start_pos

        # =====================================================
        # 2. ЕСЛИ АГЕНТ НА БАЗЕ — ЗАРЯДИЛСЯ И ПРОДОЛЖАЕТ
        # =====================================================
        if env.pos == env.start_pos:
            env.energy_system.recharge()

        # =====================================================
        # 3. ПРОВЕРЯЕМ, МОЖЕМ ЛИ ВООБЩЕ ВЕРНУТЬСЯ ДОМОЙ
        # =====================================================
        path_home = self.planner.find_path_oriented(
            env,
            env.pos,
            env.start_pos,
            memory=self.memory,
            robot_id=self.robot_id,
            blackboard=self.blackboard
        )

        if path_home is None:
            self.mode = "STUCK"
            return None

        home_cost = self.estimate_path_cost(env, path_home)

        if not env.energy_system.can_reach(home_cost, reserve=5.0):
            self.mode = "RETURN_CHARGE"
            return env.start_pos

        # =====================================================
        # 4. ВЫБИРАЕМ СЕКТОР
        # =====================================================
        sector = self.mission.current_sector(
            env,
            self.memory,
            self.sectors
        )
        self.sectors.current_sector = sector

        if sector is not None:
            claimed = self.blackboard.claim_sector(self.robot_id, sector)

            if not claimed:
                self.sectors.current_sector = None
                sector = None
        # =====================================================
        # 5. ЕСЛИ СЕКТОР НАЙДЕН — ПРОВЕРЯЕМ ЭНЕРГИЮ НА СЕКТОР
        # =====================================================
        if sector is not None:
            ok_energy, required_energy = self.energy_predictor.has_energy_to_finish_sector(
                env,
                self.planner,
                self.sectors,
                sector,
                memory=self.memory
            )

            self.last_required_energy = required_energy

            if not ok_energy:
                self.mode = "RETURN_CHARGE"
                return env.start_pos

            cabbage = self.coverage.get_next_target_hybrid(
                self.memory,
                env,
                self.sectors,
                sector,
                prev_pos=getattr(self, "prev_pos", None)
            )

        else:
            # ВАЖНО:
            # sector is None НЕ означает "ехать заряжаться"
            # это может значить, что секторный выбор не сработал
            self.last_required_energy = 0.0
            cabbage = None

        # =====================================================
        # 6. FALLBACK: ЕСЛИ В СЕКТОРЕ НЕТ ЦЕЛИ — ИЩЕМ КАПУСТУ НА ВСЕЙ КАРТЕ
        # =====================================================
        if cabbage is None:
            cabbage = self.nearest_cabbage(env)
        # =========================================
        #
        # ========================================
        if cabbage is None:
            frontier = self.frontiers.choose_frontier(
                env,
                self.memory,
                self.planner,
                self.energy_predictor
            )

            if frontier is not None:
                self.mode = "EXPLORE"
                return frontier

        # =====================================================
        # 7. ЕСЛИ КАПУСТЫ НЕТ ВООБЩЕ — ФИНИШ
        # =====================================================
        if cabbage is None:
            self.mode = "RETURN_FINISH"
            return env.start_pos

        # ===========================
        dynamic_positions = (
            env.dynamic_obstacles.positions()
            if hasattr(env, "dynamic_obstacles")
            else set()
        )

        if self.goal in dynamic_positions:
            self.path = None
            self.goal = None
            self.replan_cooldown = 0

        # =====================================================
        # 8. СТРОИМ ПУТЬ ДО КАПУСТЫ
        # =====================================================
        path_to_cabbage = self.planner.find_path_oriented(
            env,
            env.pos,
            cabbage,
            memory=self.memory,
            unknown_policy="avoid",
            robot_id=self.robot_id,
            blackboard=self.blackboard
        )

        if path_to_cabbage is None:
            # не надо сразу RETURN_CHARGE
            # возможно именно эта капуста недоступна — fallback уже был,
            # но если A* не нашёл путь, тогда возвращаемся безопасно
            self.mode = "RETURN_CHARGE"
            return env.start_pos

        to_cabbage_cost = self.estimate_path_cost(
            env,
            path_to_cabbage
        )

        # =====================================================
        # 9. ПРОВЕРЯЕМ ВОЗВРАТ ПОСЛЕ ЭТОЙ КАПУСТЫ
        # =====================================================
        path_back = self.planner.find_path_oriented(
            env,
            cabbage,
            env.start_pos,
            start_heading=env.heading,
            memory=self.memory,
            unknown_policy="avoid",
            robot_id=self.robot_id,
            blackboard=self.blackboard
        )

        if path_back is None:
            self.mode = "RETURN_CHARGE"
            return env.start_pos

        back_cost = self.estimate_path_cost(
            env,
            path_back
        )

        total_mission_cost = to_cabbage_cost + back_cost

        if not env.energy_system.can_reach(total_mission_cost, reserve=5.0):
            self.mode = "RETURN_CHARGE"
            return env.start_pos

        # =====================================================
        # 10. ВСЁ ОК → СОБИРАЕМ
        # =====================================================
        self.mode = "COLLECT"

        return cabbage

    def action_from_path(self, env, path):
        if path is None or len(path) < 2:
            return 0

        x, y = env.pos
        nx, ny = path[1]

        dx = nx - x
        dy = ny - y

        for i, (adx, ady) in enumerate(ACTIONS[:4]):
            if (adx, ady) == (dx, dy):
                return i

        return 0

    def act(self, env, temp=0):

        # =====================================================
        # 1. TEAM UPDATE
        # =====================================================
        self.blackboard.update_robot(
            self.robot_id,
            env.pos
        )

        # =====================================================
        # 2. MEMORY UPDATE
        # =====================================================
        if self.memory.map is None:
            self.memory.reset(env.grid.shape)

        self.memory.observe_local(env, radius=3)

        # shared memory
        self.blackboard.update_shared_memory(self.memory)
        self.memory = self.blackboard.sync_memory(self.memory)

        # sync current path with actual position
        self.sync_path_with_position(env)

        # =====================================================
        # 3. INIT INTERNAL STATE
        # =====================================================
        if not hasattr(self, "replan_cooldown"):
            self.replan_cooldown = 0

        if not hasattr(self, "replan_interval"):
            self.replan_interval = 8

        if not hasattr(self, "prev_pos"):
            self.prev_pos = None

        remaining = np.sum(env.grid == 1)

        env.allow_start_access = (
                self.mode == "RETURN_CHARGE"
                or (self.mode == "RETURN_FINISH" and remaining == 0)
        )

        # =====================================================
        # 4. RISK MODE
        # =====================================================
        risk_mode = self.compute_risk_mode(env)

        if risk_mode == "CAREFUL":
            current_replan_interval = 2

        elif risk_mode == "SAFE_RETURN":
            current_replan_interval = 1

        else:
            current_replan_interval = self.replan_interval

        # =====================================================
        # 5. CHECK IF REPLAN NEEDED
        # =====================================================
        need_replan = False

        if self.goal is None:
            need_replan = True

        if self.path is None or len(self.path) < 2:
            need_replan = True

        blocked = self.path_is_blocked(env)

        if blocked:
            self.recovery.report_blocked()
            need_replan = True
            self.replan_cooldown = 0


        if self.replan_cooldown <= 0:
            need_replan = True

        if self.goal is not None and env.pos == self.goal:
            need_replan = True


        dynamic_positions = (
            env.dynamic_obstacles.positions()
            if hasattr(env, "dynamic_obstacles")
            else set()
        )

        if self.goal in dynamic_positions:
            self.goal = None
            self.path = None
            self.replan_cooldown = 0
            need_replan = True

        # =====================================================
        # 6. REPLAN
        # =====================================================
        if need_replan:

            self.goal = self.choose_goal(env)

            # unknown policy
            if risk_mode == "SAFE_RETURN":
                unknown_policy = "avoid"

            elif self.mode == "EXPLORE":
                unknown_policy = "explore"

            elif self.mode in ["RETURN_CHARGE", "RETURN_FINISH"]:
                unknown_policy = "avoid"

            else:
                unknown_policy = "allow"

            # build path
            if self.goal is not None:

                self.path = self.planner.find_path_oriented(
                    env,
                    env.pos,
                    self.goal,
                    memory=self.memory,
                    unknown_policy=unknown_policy,
                    robot_id=self.robot_id,
                    blackboard=self.blackboard
                )

            else:
                self.path = None

            if self.path is None:
                self.recovery.report_no_path()
            else:
                self.recovery.clear_soft_failures()

            self.replan_cooldown = current_replan_interval

        else:
            self.replan_cooldown -= 1

        # =====================================================
        # 7. ANTI BACKTRACK
        # =====================================================
        if (
                self.prev_pos is not None
                and self.path is not None
                and len(self.path) >= 2
                and self.path[1] == self.prev_pos
                and not self.path_is_blocked(env)
        ):

            self.path = None
            self.replan_cooldown = 0

            self.goal = self.choose_goal(env)

            if self.goal is not None:
                self.path = self.planner.find_path_oriented(
                    env,
                    env.pos,
                    self.goal,
                    memory=self.memory,
                    unknown_policy="allow",
                    robot_id=self.robot_id,
                    blackboard=self.blackboard
                )

        # =====================================================
        # 8. ACTION SELECTION / RECOVERY
        # =====================================================
        recovery_mode = self.recovery.choose_recovery_mode()

        if recovery_mode == "WAIT":
            action = self.safe_wait_action(env)

        elif recovery_mode == "BACK_OFF":
            action = self.backoff_action(env)

        elif recovery_mode == "EXPLORE_ALT":
            self.goal = None
            self.path = None
            self.replan_cooldown = 0
            action = self.safe_detour_action(env)

        else:
            if self.path is not None and len(self.path) >= 2:
                action = self.action_from_path(
                    env,
                    self.path
                )
            else:
                action = self.safe_detour_action(env)

        # =====================================================
        # 9. METRICS
        # =====================================================
        sector = self.sectors.current_sector

        if not hasattr(self, "last_sector"):
            self.last_sector = None

        if not hasattr(self, "sector_switches"):
            self.sector_switches = 0

        if sector is not None and sector != self.last_sector:
            self.sector_switches += 1
            self.last_sector = sector

        total = np.sum(env.initial_grid == 1)
        remaining = np.sum(env.grid == 1)
        collected = total - remaining

        energy_used = getattr(env, "energy_used", 0.0)

        energy_per_cabbage = (
                energy_used / max(1, collected)
        )

        if hasattr(env, "visit_count"):

            overlap_cells = int(
                np.sum(env.visit_count > 1)
            )

            visited_cells = int(
                np.sum(env.visit_count > 0)
            )

            overlap_rate = (
                    overlap_cells / max(1, visited_cells)
            )

        else:
            overlap_cells = 0
            overlap_rate = 0.0

        total_turns = getattr(env, "total_turns", 0)

        required_energy = getattr(
            self,
            "last_required_energy",
            0.0
        )

        energy_margin = (
                env.energy_system.energy
                - required_energy
        )

        frontiers = self.memory.frontier_cells()


        # =====================================================
        # 10. DEBUG
        # =====================================================
        debug = {

            # state
            "mode": self.mode,
            "risk_mode": risk_mode,

            # navigation
            "goal": self.goal,
            "path": self.path,
            "need_replan": need_replan,
            "replan_cooldown": self.replan_cooldown,

            # sectors
            "sector": self.sectors.current_sector,
            "sector_h": self.sectors.sector_h,
            "sector_w": self.sectors.sector_w,
            "sector_switches": self.sector_switches,

            # frontiers
            "frontiers": frontiers,
            "frontier_count": len(frontiers),

            "frontier_clusters": getattr(
                self.frontiers,
                "frontier_clusters",
                []
            ),

            "frontier_target": getattr(
                self.frontiers,
                "selected_frontier",
                None
            ),

            # memory
            "memory_map": self.memory.map.copy(),
            "memory_seen": self.memory.seen.copy(),

            "memory_coverage":
                self.memory.coverage_rate(),

            "memory_overlap":
                self.memory.visited_overlap_rate(),

            # energy
            "energy": env.energy_system.energy,
            "max_energy": env.energy_system.max_energy,
            "energy_used": energy_used,
            "energy_per_cabbage": energy_per_cabbage,
            "required_energy": required_energy,
            "energy_margin": energy_margin,

            # robot
            "knife_on": env.knife_on,
            "heading": env.heading,

            # stats
            "total_turns": total_turns,
            "overlap_cells": overlap_cells,
            "overlap_rate": overlap_rate,

            # multi-agent
            "robot_id": self.robot_id,

            "claimed_sectors":
                dict(self.blackboard.claimed_sectors),

            "robot_positions":
                dict(self.blackboard.robot_positions),
            "opportunistic_sector":
                getattr(self, "last_opportunistic_sector", None),

            "dynamic_predictions":
                dict(env.dynamic_obstacles.predicted_positions())
                if hasattr(env, "dynamic_obstacles")
                else {},
            "dynamic_traffic": self.memory.dynamic_traffic.copy(),
            "recovery_mode": self.recovery.recovery_mode,
            "no_path_counter": self.recovery.no_path_counter,
            "blocked_counter": self.recovery.blocked_counter,
            "recovery_counts": dict(self.recovery.recovery_counts),
        }

        # =====================================================
        # 11. SAVE POSITION
        # =====================================================
        self.prev_pos = env.pos
        self.recovery.update(env, debug)

        return action, debug

    def estimate_path_cost(self, env, path, start_heading=None):
        if path is None or len(path) < 2:
            return 0.0

        heading = env.heading if start_heading is None else start_heading
        cost = 0.0

        for i in range(len(path) - 1):
            x, y = path[i]
            nx, ny = path[i + 1]

            dx = nx - x
            dy = ny - y

            # движение
            cost += MOVE_COST

            # поворот
            target_heading = heading

            for h, (hx, hy) in enumerate(DIRECTIONS):
                if (dx, dy) == (hx, hy):
                    target_heading = h
                    break

            diff = abs(target_heading - heading)
            diff = min(diff, 4 - diff)

            cost += diff * TURN_COST
            heading = target_heading

            # нож только если следующая клетка с капустой
            if env.grid[nx][ny] == 1:
                cost += CUT_COST

        return cost

    def path_is_blocked(self, env):
        if self.path is None or len(self.path) < 2:
            return False

        next_pos = self.path[1]

        dynamic_positions = (
            env.dynamic_obstacles.positions()
            if hasattr(env, "dynamic_obstacles")
            else set()
        )

        if next_pos in dynamic_positions:
            return True

        if next_pos in env.obstacles:
            return True

        return False

    def sync_path_with_position(self, env):
        if self.path is None:
            return

        if env.pos in self.path:
            idx = self.path.index(env.pos)
            self.path = self.path[idx:]
        else:
            self.path = None

    def safe_detour_action(self, env):
        dynamic_positions = (
            env.dynamic_obstacles.positions()
            if hasattr(env, "dynamic_obstacles")
            else set()
        )

        x, y = env.pos
        h, w = env.grid.shape

        for a, (dx, dy) in enumerate(ACTIONS[:4]):
            nx = max(0, min(h - 1, x + dx))
            ny = max(0, min(w - 1, y + dy))
            np_ = (nx, ny)

            if np_ in env.obstacles:
                continue
            if np_ in dynamic_positions:
                continue
            if np_ == getattr(self, "prev_pos", None):
                continue

            return a

        return 0

    def compute_risk_mode(self, env):
        dynamic_positions = (
            env.dynamic_obstacles.positions()
            if hasattr(env, "dynamic_obstacles")
            else set()
        )

        x, y = env.pos

        min_dist = 999

        for ox, oy in dynamic_positions:
            d = abs(x - ox) + abs(y - oy)
            min_dist = min(min_dist, d)

        energy_ratio = env.energy_system.energy / env.energy_system.max_energy

        if self.mode in ["RETURN_CHARGE", "RETURN_FINISH"] and energy_ratio < 0.3:
            return "SAFE_RETURN"

        if min_dist <= 2:
            return "CAREFUL"

        return "NORMAL"

    def find_opportunistic_sector(self, env, current_sector):
        energy = env.energy_system.energy
        self.last_opportunistic_sector = None
        best_sector = None
        best_score = -1e9

        for sector_id in self.sectors.all_sector_ids(self.memory):
            if sector_id == current_sector:
                continue

            cab_count = self.sectors.sector_cabbages(self.memory, sector_id)

            if cab_count < OPPORTUNISTIC_MIN_CABBAGES:
                continue

            center = self.sectors.sector_center(sector_id, self.memory.map.shape)

            path_to = self.planner.find_path_oriented(
                env,
                env.pos,
                center,
                memory=self.memory,
                unknown_policy="allow",
                robot_id=self.robot_id,
                blackboard=self.blackboard
            )

            if path_to is None:
                continue

            path_home = self.planner.find_path_oriented(
                env,
                center,
                env.start_pos,
                memory=self.memory,
                unknown_policy="avoid",
                robot_id=self.robot_id,
                blackboard=self.blackboard
            )

            if path_home is None:
                continue

            to_cost = self.estimate_path_cost(env, path_to)
            home_cost = self.estimate_path_cost(env, path_home)

            extra_cost = to_cost + home_cost

            score = cab_count * 10 - extra_cost
            print(
                "OPP",
                "sector", sector_id,
                "cab", cab_count,
                "extra", round(extra_cost, 1),
                "energy", round(energy, 1),
                "score", round(score, 1)
            )
            if extra_cost > OPPORTUNISTIC_MAX_EXTRA_COST:
                continue

            if energy - extra_cost < OPPORTUNISTIC_RETURN_MARGIN:
                continue


            if score > best_score:
                best_score = score
                best_sector = sector_id

        self.last_opportunistic_sector = best_sector

        return best_sector

    def safe_wait_action(self, env):
        """
        Пока у нас нет action='wait', поэтому выбираем самый безопасный малый detour.
        Позже можно добавить отдельное действие WAIT.
        """
        return WAIT_ACTION

    def backoff_action(self, env):
        """
        Пытается выйти из локального застревания:
        предпочтительно не идти в последнюю часто посещённую клетку.
        """
        dynamic_positions = (
            env.dynamic_obstacles.positions()
            if hasattr(env, "dynamic_obstacles")
            else set()
        )

        x, y = env.pos

        best_action = None
        best_score = 1e18

        for a, (dx, dy) in enumerate(ACTIONS[:4]):
            nx = max(0, min(env.grid.shape[0] - 1, x + dx))
            ny = max(0, min(env.grid.shape[1] - 1, y + dy))

            p = (nx, ny)

            if p in env.obstacles:
                continue

            if p in dynamic_positions:
                continue

            score = 0.0

            if hasattr(env, "visit_count"):
                score += env.visit_count[nx, ny] * 2.0

            if p == getattr(self, "prev_pos", None):
                score += 5.0

            if p == env.start_pos and np.sum(env.grid == 1) > 0:
                score += 10.0

            if score < best_score:
                best_score = score
                best_action = a

        if best_action is None:
            return self.safe_detour_action(env)

        return best_action