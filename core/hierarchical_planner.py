import heapq


class HierarchicalPlanner:
    def __init__(self, sector_manager, low_level_planner):
        self.sectors = sector_manager
        self.low = low_level_planner

    def sector_of_pos(self, pos):
        x, y = pos
        return (
            x // self.sectors.sector_h,
            y // self.sectors.sector_w
        )

    def sector_center(self, sector_id, shape):
        x1, x2, y1, y2 = self.sectors.get_sector_bounds(
            sector_id,
            shape
        )

        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )

    def sector_neighbors(self, sector_id, memory):
        sx, sy = sector_id
        candidates = [
            (sx + 1, sy),
            (sx - 1, sy),
            (sx, sy + 1),
            (sx, sy - 1),
        ]

        valid = set(self.sectors.all_sector_ids(memory))

        return [
            s for s in candidates
            if s in valid
        ]

    def sector_cost(self, a, b, memory):
        ax, ay = self.sector_center(a, memory.map.shape)
        bx, by = self.sector_center(b, memory.map.shape)

        return abs(ax - bx) + abs(ay - by)

    def find_sector_route(self, start_sector, goal_sector, memory):
        if start_sector == goal_sector:
            return [start_sector]

        open_set = []
        heapq.heappush(open_set, (0, start_sector))

        came_from = {}
        g = {start_sector: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_sector:
                break

            for nb in self.sector_neighbors(current, memory):
                cost = self.sector_cost(current, nb, memory)
                new_g = g[current] + cost

                if nb not in g or new_g < g[nb]:
                    g[nb] = new_g
                    came_from[nb] = current

                    h = self.sector_cost(nb, goal_sector, memory)
                    heapq.heappush(open_set, (new_g + h, nb))

        if goal_sector not in came_from and start_sector != goal_sector:
            return None

        route = [goal_sector]
        cur = goal_sector

        while cur != start_sector:
            cur = came_from[cur]
            route.append(cur)

        route.reverse()
        return route

    def build_waypoints(self, sector_route, memory):
        if sector_route is None:
            return []

        return [
            self.sector_center(s, memory.map.shape)
            for s in sector_route[1:]
        ]

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
        if memory is None or memory.map is None:
            return self.low.find_path_oriented(
                env,
                start,
                goal,
                start_heading=start_heading,
                memory=memory,
                unknown_policy=unknown_policy,
                robot_id=robot_id,
                blackboard=blackboard
            )

        start_sector = self.sector_of_pos(start)
        goal_sector = self.sector_of_pos(goal)

        route = self.find_sector_route(
            start_sector,
            goal_sector,
            memory
        )

        if route is None:
            return self.low.find_path_oriented(
                env,
                start,
                goal,
                start_heading=start_heading,
                memory=memory,
                unknown_policy=unknown_policy,
                robot_id=robot_id,
                blackboard=blackboard
            )

        waypoints = self.build_waypoints(route, memory)
        waypoints.append(goal)

        full_path = []
        current_start = start
        heading = start_heading

        for wp in waypoints:
            part = self.low.find_path_oriented(
                env,
                current_start,
                wp,
                start_heading=heading,
                memory=memory,
                unknown_policy=unknown_policy,
                robot_id=robot_id,
                blackboard=blackboard
            )

            if part is None:
                return None

            if len(full_path) == 0:
                full_path.extend(part)
            else:
                full_path.extend(part[1:])

            current_start = wp

        return full_path