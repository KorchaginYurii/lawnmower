class PortalPlanner:
    def __init__(self, sector_manager, low_level_planner):
        self.sectors = sector_manager
        self.low = low_level_planner

    def sector_of_pos(self, pos):
        x, y = pos
        return x // self.sectors.sector_h, y // self.sectors.sector_w

    def sector_center(self, sector_id, shape):
        x1, x2, y1, y2 = self.sectors.get_sector_bounds(sector_id, shape)
        return (x1 + x2) // 2, (y1 + y2) // 2

    def find_portal_between(self, a, b, memory):
        ax1, ax2, ay1, ay2 = self.sectors.get_sector_bounds(a, memory.map.shape)
        bx1, bx2, by1, by2 = self.sectors.get_sector_bounds(b, memory.map.shape)

        candidates = []

        # vertical neighbors
        if ax2 == bx1 or bx2 == ax1:
            row = ax2 - 1 if ax2 == bx1 else bx2 - 1
            next_row = row + 1
            y_start = max(ay1, by1)
            y_end = min(ay2, by2)

            for y in range(y_start, y_end):
                if memory.map[row, y] != -1 and memory.map[next_row, y] != -1:
                    candidates.append((row, y))
                    candidates.append((next_row, y))

        # horizontal neighbors
        if ay2 == by1 or by2 == ay1:
            col = ay2 - 1 if ay2 == by1 else by2 - 1
            next_col = col + 1
            x_start = max(ax1, bx1)
            x_end = min(ax2, bx2)

            for x in range(x_start, x_end):
                if memory.map[x, col] != -1 and memory.map[x, next_col] != -1:
                    candidates.append((x, col))
                    candidates.append((x, next_col))

        if not candidates:
            return None

        # пока берём середину свободной границы
        return candidates[len(candidates) // 2]

    def sector_neighbors(self, sector_id, memory):
        sx, sy = sector_id
        possible = [
            (sx + 1, sy),
            (sx - 1, sy),
            (sx, sy + 1),
            (sx, sy - 1),
        ]

        valid = set(self.sectors.all_sector_ids(memory))
        return [s for s in possible if s in valid]

    def sector_route_greedy(self, start_sector, goal_sector, memory):
        if start_sector == goal_sector:
            return [start_sector]

        route = [start_sector]
        current = start_sector
        visited = set()

        while current != goal_sector:
            visited.add(current)

            neighbors = [
                s for s in self.sector_neighbors(current, memory)
                if s not in visited
            ]

            if not neighbors:
                return None

            gx, gy = goal_sector

            best = min(
                neighbors,
                key=lambda s: abs(s[0] - gx) + abs(s[1] - gy)
            )

            route.append(best)
            current = best

            if len(route) > len(self.sectors.all_sector_ids(memory)):
                return None

        return route

    def build_portal_waypoints(self, sector_route, memory):
        if sector_route is None or len(sector_route) < 2:
            return []

        waypoints = []

        for a, b in zip(sector_route[:-1], sector_route[1:]):
            portal = self.find_portal_between(a, b, memory)

            if portal is None:
                return None

            waypoints.append(portal)

        return waypoints

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
                env, start, goal,
                start_heading=start_heading,
                memory=memory,
                unknown_policy=unknown_policy,
                robot_id=robot_id,
                blackboard=blackboard
            )

        start_sector = self.sector_of_pos(start)
        goal_sector = self.sector_of_pos(goal)

        route = self.sector_route_greedy(
            start_sector,
            goal_sector,
            memory
        )

        waypoints = self.build_portal_waypoints(route, memory)

        if waypoints is None:
            return self.low.find_path_oriented(
                env, start, goal,
                start_heading=start_heading,
                memory=memory,
                unknown_policy=unknown_policy,
                robot_id=robot_id,
                blackboard=blackboard
            )

        waypoints.append(goal)

        full_path = []
        current = start
        heading = start_heading

        for wp in waypoints:
            part = self.low.find_path_oriented(
                env,
                current,
                wp,
                start_heading=heading,
                memory=memory,
                unknown_policy=unknown_policy,
                robot_id=robot_id,
                blackboard=blackboard
            )

            if part is None:
                return self.low.find_path_oriented(
                    env, start, goal,
                    start_heading=start_heading,
                    memory=memory,
                    unknown_policy=unknown_policy,
                    robot_id=robot_id,
                    blackboard=blackboard
                )

            if not full_path:
                full_path.extend(part)
            else:
                full_path.extend(part[1:])

            current = wp

        return full_path