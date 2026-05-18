import heapq
import numpy as np
from core.world_memory import UNKNOWN, OBSTACLE
from core.config import (
    ACTIONS,
    DIRECTIONS,
    MOVE_COST,
    TURN_COST,
    UNKNOWN_COST_AVOID,
    UNKNOWN_COST_ALLOW,
    UNKNOWN_COST_EXPLORE,
)
from core.tuning_config import runtime_config


class AStarPlanner:
    def __init__(self):
        pass

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, env, start, goal):
        if start == goal:
            return [start]

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            x, y = current
            h, w = env.grid.shape
            for dx, dy in ACTIONS[:4]:
                nx = max(0, min(h - 1, x + dx))
                ny = max(0, min(w - 1, y + dy))

                neighbor = (nx, ny)

                if neighbor in env.obstacles:
                    continue
                # старт запрещён как проходная клетка, пока капуста ещё есть
                if hasattr(env, "start_pos"):
                    remaining = np.sum(env.grid == 1)

                    if remaining > 0 and not getattr(env, "allow_start_access", False):
                        # разрешаем старт только если это начальная клетка или конечная цель
                        if neighbor == env.start_pos and neighbor != start and neighbor != goal:
                            continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    f = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return None

    def reconstruct_path(self, came_from, current):
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def direction_to_heading(self, dx, dy, current_heading):
        for i, (hx, hy) in enumerate(DIRECTIONS):
            if (dx, dy) == (hx, hy):
                return i
        return current_heading

    def turn_cost(self, from_heading, to_heading):
        diff = abs(to_heading - from_heading)
        diff = min(diff, 4 - diff)
        return diff * TURN_COST

    def find_path_oriented(
            self,
            env,
            start,
            goal,
            start_heading=None,
            memory=None,
            unknown_policy="allow",
            robot_id=None,
            blackboard=None
    ):
        if start == goal:
            return [start]

        if start_heading is None:
            start_heading = getattr(env, "heading", 0)

        start_state = (start[0], start[1], start_heading)

        open_set = []
        heapq.heappush(open_set, (0.0, start_state))

        came_from = {}
        g_score = {start_state: 0.0}

        best_goal_state = None
        #карта
        h, w = env.grid.shape

        dynamic_positions = (
            env.dynamic_obstacles.positions()
            if hasattr(env, "dynamic_obstacles")
            else set()
        )

        robot_positions = set()

        if blackboard is not None:
            robot_positions = set(blackboard.robot_positions.values())

            if robot_id is not None and robot_id in blackboard.robot_positions:
                robot_positions.discard(blackboard.robot_positions[robot_id])

        while open_set:
            _, current = heapq.heappop(open_set)

            x, y, heading = current

            if (x, y) == goal:
                best_goal_state = current
                break

            for dx, dy in ACTIONS[:4]:
                nx = max(0, min(h - 1, x + dx))
                ny = max(0, min(w - 1, y + dy))

                neighbor_pos = (nx, ny)

                if neighbor_pos in dynamic_positions:
                    continue

                if blackboard is not None:
                  if neighbor_pos in robot_positions:
                        continue

                # ===== static obstacles =====
                if memory is not None and memory.map is not None:
                    if memory.map[nx, ny] == OBSTACLE:
                        continue
                else:
                    if neighbor_pos in env.obstacles:
                        continue


                # старт нельзя использовать как проходную клетку во время сбора
                if hasattr(env, "start_pos"):
                    remaining = np.sum(env.grid == 1)

                    if remaining > 0 and not getattr(env, "allow_start_access", False):
                        if neighbor_pos == env.start_pos and neighbor_pos != start and neighbor_pos != goal:
                            continue

                target_heading = self.direction_to_heading(dx, dy, heading)

                move_cost = MOVE_COST

                turn_weight = runtime_config.get("TURN_COST_WEIGHT", 0.3)
                straight_bonus = runtime_config.get("STRAIGHT_BONUS", 0.03)
                turn_change_penalty = runtime_config.get("TURN_CHANGE_PENALTY", 0.05)

                base_turn_cost = self.turn_cost(heading, target_heading)
                rotate_cost = turn_weight * base_turn_cost

                # бонус за продолжение движения прямо
                if target_heading == heading:
                    move_cost -= straight_bonus

                # дополнительный мягкий штраф за смену направления
                if target_heading != heading:
                    rotate_cost += turn_change_penalty

                move_cost = max(0.01, move_cost)

                unknown_cost = self.memory_cell_extra_cost(
                    memory,
                    neighbor_pos,
                    unknown_policy=unknown_policy
                )

                dynamic_cost = self.dynamic_obstacle_penalty(
                    env,
                    neighbor_pos
                )

                prediction_cost = self.dynamic_prediction_penalty(env, neighbor_pos)

                memory_dynamic_cost = 0.0

                if memory is not None:
                    memory_dynamic_cost = memory.dynamic_risk(neighbor_pos) * 1.0

                traffic_cost = 0.0

                if memory is not None:
                    traffic_cost = (
                            memory.dynamic_traffic_risk(neighbor_pos)
                            * runtime_config.get("DYNAMIC_TRAFFIC_COST", 0.4)
                    )

                prediction_cost = self.dynamic_prediction_penalty(
                    env,
                    neighbor_pos
                )

                step_cost = (
                        move_cost
                        + rotate_cost
                        + unknown_cost
                        + dynamic_cost
                        + prediction_cost
                        + memory_dynamic_cost
                        + traffic_cost
                )

                neighbor_state = (nx, ny, target_heading)

                tentative_g = g_score[current] + step_cost

                if neighbor_state not in g_score or tentative_g < g_score[neighbor_state]:
                    came_from[neighbor_state] = current
                    g_score[neighbor_state] = tentative_g

                    f = tentative_g + self.heuristic(neighbor_pos, goal) * MOVE_COST
                    heapq.heappush(open_set, (f, neighbor_state))

        if best_goal_state is None:
            return None

        return self.reconstruct_oriented_path(came_from, best_goal_state)

    def reconstruct_oriented_path(self, came_from, current):
        path = [(current[0], current[1])]

        while current in came_from:
            current = came_from[current]
            path.append((current[0], current[1]))

        path.reverse()

        # убираем возможные дубли
        compact = []
        for p in path:
            if not compact or compact[-1] != p:
                compact.append(p)

        return compact

    def memory_cell_extra_cost(self, memory, pos, unknown_policy="allow"):
        if memory is None or memory.map is None:
            return 0.0

        x, y = pos

        if memory.map[x, y] != UNKNOWN:
            return 0.0

        if unknown_policy == "avoid":
            return runtime_config.get("UNKNOWN_COST_AVOID", 1.0)

        if unknown_policy == "explore":
            return runtime_config.get("UNKNOWN_COST_EXPLORE", 0.05)

        return runtime_config.get("UNKNOWN_COST_ALLOW", 0.3)

    def dynamic_obstacle_penalty(self, env, pos):
        if not hasattr(env, "dynamic_obstacles"):
            return 0.0

        dynamic_positions = env.dynamic_obstacles.positions()

        if len(dynamic_positions) == 0:
            return 0.0

        x, y = pos

        min_dist = float("inf")

        for ox, oy in dynamic_positions:
            d = abs(x - ox) + abs(y - oy)

            if d < min_dist:
                min_dist = d

        # obstacle cell
        if min_dist == 0:
            return 9999.0

        # very dangerous
        elif min_dist == 1:
            return runtime_config.get("DYNAMIC_NEAR_COST", 2.0)

        # nearby
        elif min_dist == 2:
            return runtime_config.get("DYNAMIC_MID_COST", 0.5)

        # slight discomfort
        elif min_dist == 3:
            return runtime_config.get("DYNAMIC_FAR_COST", 0.1)

        return 0.0

    def dynamic_prediction_penalty(self, env, pos):
        if not hasattr(env, "dynamic_obstacles"):
            return 0.0

        horizon = runtime_config.get("PREDICTION_HORIZON", 3)
        base_cost = runtime_config.get("PREDICTION_COST", 1.2)
        decay = runtime_config.get("PREDICTION_DECAY", 0.6)

        predicted = env.dynamic_obstacles.predicted_positions(
            horizon=horizon
        )

        if pos not in predicted:
            return 0.0

        t = predicted[pos]

        return base_cost * (decay ** (t - 1))