import pygame
import numpy as np

MARGIN = 1

WHITE = (235, 235, 235)
BG = (18, 18, 18)

EMPTY = (235, 235, 235)
GRASS = (70, 200, 70)
CUT = (160, 220, 150)
OBSTACLE = (25, 25, 25)
BUFFER = (100, 80, 60)

ROBOT = (240, 50, 50)
BASE = (50, 80, 255)
GOAL = (255, 220, 40)
PATH = (0, 200, 255)

LANE_ACTIVE = (80, 160, 255, 90)
LANE_DONE = (180, 180, 180, 70)
LANE_BLOCKED = (255, 60, 60, 80)


class LawnRenderer:
    def __init__(
        self,
        map_shape,
        window_w=1400,
        window_h=850,
        hud_w=330,
        hud_h=100,
        min_cell=8,
        max_cell=42,
    ):
        pygame.init()

        self.h, self.w = map_shape
        self.hud_w = hud_w
        self.hud_h = hud_h

        usable_w = window_w - hud_w
        usable_h = window_h - hud_h

        self.cell = max(
            min_cell,
            min(max_cell, usable_w // self.w, usable_h // self.h)
        )

        self.map_w = self.w * self.cell
        self.map_h = self.h * self.cell

        self.screen = pygame.display.set_mode(
            (self.map_w + hud_w, self.map_h + hud_h)
        )

        pygame.display.set_caption("LawnMower Bot")

        self.font = pygame.font.SysFont("consolas", 17)
        self.small_font = pygame.font.SysFont("consolas", 14)

        self.ui_items = [
            {"name": "Path", "key": "show_path", "value": True},
            {"name": "Goal", "key": "show_goal", "value": True},
            {"name": "Visit Heatmap", "key": "show_visit", "value": False},
            {"name": "Cut Heatmap", "key": "show_cut", "value": False},
            {"name": "Lane Memory", "key": "show_lanes", "value": True},
        ]

        self.show_path = True
        self.show_goal = True
        self.show_visit = False
        self.show_cut = False
        self.show_lanes = True

    def draw(self, env, debug=None):
        debug = debug or {}

        self.screen.fill(BG)

        self.draw_map(env)

        if self.show_lanes:
            self.draw_lane_memory(env, debug)

        if self.show_visit and hasattr(env, "visit_count"):
            self.draw_heatmap(env.visit_count, color=(255, 80, 60))

        if self.show_cut and hasattr(env, "cut_count"):
            self.draw_heatmap(env.cut_count, color=(60, 180, 255))

        if self.show_path:
            self.draw_path(debug.get("path") or debug.get("return_path"))

        if self.show_goal:
            self.draw_goal(debug.get("goal"))

        self.draw_base(env)
        self.draw_robot(env, debug)

        self.draw_side_panel(env, debug)
        self.draw_bottom_hud(env, debug)

        pygame.display.flip()

    def draw_map(self, env):
        for x in range(env.grid.shape[0]):
            for y in range(env.grid.shape[1]):
                v = env.grid[x, y]

                if v == 1:
                    color = GRASS
                elif v == 2:
                    color = CUT
                elif v == -1:
                    color = OBSTACLE
                elif v == 3:
                    color = BUFFER
                else:
                    color = EMPTY

                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        y * self.cell,
                        x * self.cell,
                        self.cell - MARGIN,
                        self.cell - MARGIN,
                    ),
                )

    def draw_robot(self, env, debug=None):
        debug = debug or {}

        x, y = env.pos

        cell = self.cell

        robot_size_m = getattr(env, "robot_size_m", 0.5)
        cell_size_m = getattr(env, "cell_size_m", 0.25)

        body_cells = max(1, int(round(robot_size_m / cell_size_m)))

        knife_radius_cells = max(
            1,
            int(round((robot_size_m / 2.0) / cell_size_m))
        )

        collision_radius_cells = max(
            1,
            int(round((robot_size_m / 2.0) / cell_size_m))
        )

        cx = y * cell + cell // 2
        cy = x * cell + cell // 2

        knife_on = debug.get(
            "knife_on_real",
            getattr(env, "knife_on", True)
        )

        mode = debug.get("mode", "")

        if mode == "RETURN_HOME":
            body_color = (80, 160, 255)
        elif mode == "RECOVERY":
            body_color = (255, 220, 60)
        elif mode == "FINISHED":
            body_color = (60, 220, 60)
        else:
            body_color = ROBOT

        # =========================
        # KNIFE AREA
        # =========================

        knife_radius_px = knife_radius_cells * cell

        if knife_on:
            pygame.draw.circle(
                self.screen,
                (255, 80, 80, 70),
                (cx, cy),
                knife_radius_px,
            )
        else:
            pygame.draw.circle(
                self.screen,
                (180, 180, 180),
                (cx, cy),
                knife_radius_px,
                1,
            )

        # =========================
        # COLLISION RADIUS
        # =========================

        collision_radius_px = collision_radius_cells * cell

        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (cx, cy),
            collision_radius_px,
            1,
        )

        # =========================
        # BODY
        # =========================

        half_px = (body_cells * cell) // 2

        body_rect = pygame.Rect(
            cx - half_px,
            cy - half_px,
            body_cells * cell,
            body_cells * cell,
        )

        if knife_on:
            pygame.draw.rect(
                self.screen,
                body_color,
                body_rect,
            )
        else:
            pygame.draw.rect(
                self.screen,
                body_color,
                body_rect,
                3,
            )

        # =========================
        # CENTER POINT
        # =========================

        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (cx, cy),
            3,
        )

        # =========================
        # HEADING ARROW
        # =========================

        heading = getattr(env, "heading", 1)
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = dirs[heading]

        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (cx, cy),
            (
                cx + dy * cell * 1.2,
                cy + dx * cell * 1.2,
            ),
            3,
        )

    def draw_base(self, env):
        sx, sy = env.start_pos

        pygame.draw.rect(
            self.screen,
            BASE,
            (
                sy * self.cell,
                sx * self.cell,
                self.cell,
                self.cell,
            ),
            4,
        )

    def draw_goal(self, goal):
        if goal is None:
            return

        gx, gy = goal

        pygame.draw.rect(
            self.screen,
            GOAL,
            (
                gy * self.cell + 3,
                gx * self.cell + 3,
                self.cell - 6,
                self.cell - 6,
            ),
            3,
        )

    def draw_path(self, path):
        if not path:
            return

        points = []

        for x, y in path:
            points.append(
                (
                    y * self.cell + self.cell // 2,
                    x * self.cell + self.cell // 2,
                )
            )

        if len(points) >= 2:
            pygame.draw.lines(
                self.screen,
                PATH,
                False,
                points,
                2,
            )

        for p in points:
            pygame.draw.circle(self.screen, PATH, p, 3)

    def draw_heatmap(self, arr, color):
        max_v = np.max(arr)

        if max_v <= 0:
            return

        overlay = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)

        r, g, b = color

        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                v = arr[x, y]

                if v <= 0:
                    continue

                ratio = min(1.0, float(v) / max_v)
                alpha = int(40 + 160 * ratio)

                pygame.draw.rect(
                    overlay,
                    (r, g, b, alpha),
                    (
                        y * self.cell,
                        x * self.cell,
                        self.cell,
                        self.cell,
                    ),
                )

        self.screen.blit(overlay, (0, 0))

    def draw_lane_memory(self, env, debug):
        lane = debug.get("lane_memory", None)

        if not lane:
            return

        orientation = lane.get("orientation", "horizontal")
        active = lane.get("active_lane", None)

        completed = lane.get("completed_lane_ids", [])
        blocked = lane.get("blocked_lane_ids", [])

        overlay = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)

        if orientation == "horizontal":
            for lid in completed:
                self.draw_lane_overlay_row(overlay, lid, LANE_DONE)

            for lid in blocked:
                self.draw_lane_overlay_row(overlay, lid, LANE_BLOCKED)

            if active is not None:
                self.draw_lane_overlay_row(overlay, active, LANE_ACTIVE)

        else:
            for lid in completed:
                self.draw_lane_overlay_col(overlay, lid, LANE_DONE)

            for lid in blocked:
                self.draw_lane_overlay_col(overlay, lid, LANE_BLOCKED)

            if active is not None:
                self.draw_lane_overlay_col(overlay, active, LANE_ACTIVE)

        self.screen.blit(overlay, (0, 0))

    def draw_lane_overlay_row(self, overlay, row, color):
        if row is None:
            return

        pygame.draw.rect(
            overlay,
            color,
            (
                0,
                row * self.cell,
                self.map_w,
                self.cell,
            ),
        )

    def draw_lane_overlay_col(self, overlay, col, color):
        if col is None:
            return

        pygame.draw.rect(
            overlay,
            color,
            (
                col * self.cell,
                0,
                self.cell,
                self.map_h,
            ),
        )

    def draw_side_panel(self, env, debug):
        x = self.map_w + 12
        y = 15

        for item in self.ui_items:
            setattr(self, item["key"], item["value"])

            rect = pygame.Rect(x, y, 15, 15)
            pygame.draw.rect(self.screen, (210, 210, 210), rect, 2)

            if item["value"]:
                pygame.draw.line(self.screen, (0, 255, 0), (x, y), (x + 15, y + 15), 2)
                pygame.draw.line(self.screen, (0, 255, 0), (x + 15, y), (x, y + 15), 2)

            item["rect"] = rect
            self.text(item["name"], x + 24, y - 2)
            y += 24

        y += 10
        self.text("LEGEND", x, y, (255, 230, 120))
        y += 25

        legend = [
            ("Uncut grass", GRASS),
            ("Cut grass", CUT),
            ("Obstacle", OBSTACLE),
            ("Safety buffer", BUFFER),
            ("Robot", ROBOT),
            ("Base", BASE),
            ("Goal", GOAL),
            ("Path", PATH),
            ("Knife area", (255, 80, 80)),
            ("Collision radius", (255, 255, 255)),
        ]

        for name, color in legend:
            pygame.draw.rect(self.screen, color, (x, y, 16, 16))
            self.text(name, x + 24, y - 2)
            y += 22

        lane = debug.get("lane_memory", {})

        y += 12
        self.text("LANE MEMORY", x, y, (255, 230, 120))
        y += 24

        for key in [
            "orientation",
            "active_lane",
            "completed_lanes",
            "blocked_lanes",
            "total_lanes",
            "lane_progress",
        ]:
            if key in lane:
                self.text(f"{key}: {lane[key]}", x, y)
                y += 19

    def draw_bottom_hud(self, env, debug):
        y = self.map_h + 8
        x = 10
        gap = 260
        row_h = 22

        def item(col, row, text, color=WHITE):
            self.text(text, x + col * gap, y + row * row_h, color)

        coverage = env.coverage_rate()
        overlap = env.overlap_rate()

        energy = getattr(env, "energy", 0.0)
        max_energy = getattr(env, "max_energy", 100.0)
        energy_pct = energy / max(1e-9, max_energy) * 100.0

        cut = env.cut_grass()
        total = env.total_grass()
        remaining = env.remaining_grass()

        knife = "ON" if debug.get("knife_on_real", getattr(env, "knife_on", False)) else "OFF"

        item(0, 0, f"STEP {env.steps}")
        item(1, 0, f"MODE {debug.get('mode', '-')}")
        item(2, 0, f"STRIP {debug.get('strip_mode', '-')}")
        item(3, 0, f"ENERGY {energy_pct:.1f}%")
        item(4, 0, f"GRASS {cut}/{total}")

        item(0, 1, f"COVER {coverage:.3f}")
        item(1, 1, f"OVERLAP {overlap:.3f}")
        item(2, 1, f"TURNS {env.total_turns}")
        item(3, 1, f"REMAIN {remaining}")
        item(4, 1, f"RECH {getattr(env, 'recharge_count', 0)}")

        item(0, 2, f"DIR {debug.get('strip_direction', '-')}")
        item(1, 2, f"GOAL {debug.get('goal', '-')}")
        item(2, 2, f"KNIFE {knife}")
        item(3, 2, f"CUT_LAST {debug.get('cut_cells_last', 0)}")
        item(4, 2, f"CELL {debug.get('cell_under_robot', '-')}")

        item(0, 3, f"PHASE {debug.get('adaptive_phase', '-')}")
        item(1, 3, f"VW {debug.get('adaptive_visit_weight', 0):.2f}")
        item(2, 3, f"TW {debug.get('adaptive_cell_traffic_weight', 0):.2f}")
        item(3, 3, f"CW {debug.get('adaptive_cut_weight', 0):.2f}")

    def handle_mouse(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos

            for item in self.ui_items:
                if "rect" in item and item["rect"].collidepoint(mx, my):
                    item["value"] = not item["value"]

    def text(self, text, x, y, color=WHITE):
        surf = self.font.render(str(text), True, color)
        self.screen.blit(surf, (x, y))