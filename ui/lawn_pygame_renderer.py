import pygame
import numpy as np

MARGIN = 1

COLOR_BG = (18, 18, 18)
COLOR_EMPTY = (55, 55, 55)
COLOR_GRASS = (70, 190, 70)
COLOR_CUT = (170, 220, 150)
COLOR_OBSTACLE = (25, 25, 25)
COLOR_BUFFER = (90, 70, 60)
COLOR_ROBOT = (240, 60, 60)
COLOR_BASE = (70, 120, 255)
COLOR_GOAL = (255, 210, 50)
COLOR_PATH = (0, 190, 255)
COLOR_TEXT = (235, 235, 235)


class LawnRenderer:
    def __init__(
        self,
        map_shape,
        window_w=1600,
        window_h=900,
        hud_w=330,
        hud_h=100,
        min_cell=3,
        max_cell=20,
    ):
        pygame.init()

        self.h, self.w = map_shape
        self.window_w = window_w
        self.window_h = window_h
        self.hud_w = hud_w
        self.hud_h = hud_h

        usable_w = window_w - hud_w
        usable_h = window_h - hud_h

        cell_w = usable_w // self.w
        cell_h = usable_h // self.h

        self.cell = max(
            min_cell,
            min(max_cell, min(cell_w, cell_h))
        )

        self.map_w = self.w * self.cell
        self.map_h = self.h * self.cell

        self.screen = pygame.display.set_mode(
            (self.map_w + hud_w, self.map_h + hud_h)
        )

        pygame.display.set_caption("LawnMower Bot Viewer")

        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)

        self.show_path = True
        self.show_goal = True
        self.show_visit_heatmap = False
        self.show_cut_heatmap = False
        self.show_lane_info = True

        self.ui_items = [
            {"name": "Path", "key": "show_path", "value": True},
            {"name": "Goal", "key": "show_goal", "value": True},
            {"name": "Visit Heatmap", "key": "show_visit_heatmap", "value": False},
            {"name": "Cut Heatmap", "key": "show_cut_heatmap", "value": False},
            {"name": "Lane Info", "key": "show_lane_info", "value": True},
        ]

    def draw(self, lawn_env, debug=None):
        self.screen.fill(COLOR_BG)

        self.draw_map(lawn_env)

        if self.show_visit_heatmap and hasattr(lawn_env, "visit_count"):
            self.draw_heatmap(lawn_env.visit_count, color_mode="visit")

        if self.show_cut_heatmap and hasattr(lawn_env, "cut_count"):
            self.draw_heatmap(lawn_env.cut_count, color_mode="cut")

        if debug and self.show_path:
            self.draw_path(debug.get("path") or debug.get("return_path"))

        if debug and self.show_goal:
            self.draw_goal(debug.get("goal"))

        self.draw_base(lawn_env)
        self.draw_robot(lawn_env)

        self.draw_side_panel(lawn_env, debug or {})
        self.draw_bottom_hud(lawn_env, debug or {})

        pygame.display.flip()

    def draw_map(self, env):
        for x in range(env.grid.shape[0]):
            for y in range(env.grid.shape[1]):
                val = env.grid[x, y]

                if val == 1:
                    color = COLOR_GRASS
                elif val == 2:
                    color = COLOR_CUT
                elif val == -1:
                    color = COLOR_OBSTACLE
                elif val == 3:
                    color = COLOR_BUFFER
                else:
                    color = COLOR_EMPTY

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

    def draw_robot(self, env):
        x, y = env.pos

        rect = pygame.Rect(
            y * self.cell,
            x * self.cell,
            self.cell,
            self.cell,
        )

        pygame.draw.rect(self.screen, COLOR_ROBOT, rect, 3)

        cx = y * self.cell + self.cell // 2
        cy = x * self.cell + self.cell // 2

        pygame.draw.circle(
            self.screen,
            COLOR_ROBOT,
            (cx, cy),
            max(3, self.cell // 3),
        )

        heading = getattr(env, "heading", 1)
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][heading]

        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (cx, cy),
            (
                cx + dy * self.cell,
                cy + dx * self.cell,
            ),
            2,
        )

    def draw_base(self, env):
        sx, sy = env.start_pos

        pygame.draw.rect(
            self.screen,
            COLOR_BASE,
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
            COLOR_GOAL,
            (
                gy * self.cell + 2,
                gx * self.cell + 2,
                self.cell - 4,
                self.cell - 4,
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
                COLOR_PATH,
                False,
                points,
                2,
            )

        for p in points:
            pygame.draw.circle(self.screen, COLOR_PATH, p, 3)

    def draw_heatmap(self, arr, color_mode="visit"):
        max_v = np.max(arr)

        if max_v <= 0:
            return

        overlay = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)

        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                v = arr[x, y]

                if v <= 0:
                    continue

                ratio = min(1.0, float(v) / max_v)
                alpha = int(50 + 150 * ratio)

                if color_mode == "visit":
                    color = (255, 80, 60, alpha)
                else:
                    color = (60, 180, 255, alpha)

                pygame.draw.rect(
                    overlay,
                    color,
                    (
                        y * self.cell,
                        x * self.cell,
                        self.cell,
                        self.cell,
                    ),
                )

        self.screen.blit(overlay, (0, 0))

    def draw_side_panel(self, env, debug):
        x = self.map_w + 12
        y = 16

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
        y += 24

        legend = [
            ("Uncut grass", COLOR_GRASS),
            ("Cut grass", COLOR_CUT),
            ("Obstacle", COLOR_OBSTACLE),
            ("Safety buffer", COLOR_BUFFER),
            ("Robot", COLOR_ROBOT),
            ("Base", COLOR_BASE),
            ("Goal", COLOR_GOAL),
            ("Path", COLOR_PATH),
        ]

        for name, color in legend:
            pygame.draw.rect(self.screen, color, (x, y, 16, 16))
            self.text(name, x + 24, y - 1)
            y += 22

        if self.show_lane_info:
            y += 10
            self.text("LANE MEMORY", x, y, (255, 230, 120))
            y += 22

            lane = debug.get("lane_memory", {})
            for key in [
                "orientation",
                "active_lane",
                "completed_lanes",
                "blocked_lanes",
                "total_lanes",
                "lane_progress",
            ]:
                if key in lane:
                    self.text(f"{key}: {lane[key]}", x, y, (220, 220, 220))
                    y += 18

    def draw_bottom_hud(self, env, debug):
        y = self.map_h + 8
        x = 10
        gap = 250
        row_h = 22

        def item(col, row, text, color=COLOR_TEXT):
            self.text(text, x + col * gap, y + row * row_h, color)

        coverage = env.coverage_rate() if hasattr(env, "coverage_rate") else 0.0
        overlap = env.overlap_rate() if hasattr(env, "overlap_rate") else 0.0

        energy = getattr(env, "energy", 0.0)
        max_energy = getattr(env, "max_energy", 100.0)
        energy_pct = energy / max(1e-9, max_energy) * 100.0

        item(0, 0, f"STEP {getattr(env, 'steps', 0)}")
        item(1, 0, f"MODE {debug.get('mode', '-')}")
        item(2, 0, f"STRIP {debug.get('strip_mode', '-')}")
        item(3, 0, f"ENERGY {energy_pct:.1f}%")

        item(0, 1, f"COVERAGE {coverage:.3f}")
        item(1, 1, f"OVERLAP {overlap:.3f}")
        item(2, 1, f"TURNS {getattr(env, 'total_turns', 0)}")
        item(3, 1, f"RECH {getattr(env, 'recharge_count', 0)}")

        item(0, 2, f"REMAIN {env.remaining_grass() if hasattr(env, 'remaining_grass') else '-'}")
        item(1, 2, f"CUT {env.cut_grass() if hasattr(env, 'cut_grass') else '-'}")
        item(2, 2, f"DIR {debug.get('strip_direction', '-')}")
        item(3, 2, f"GOAL {debug.get('goal', '-')}")

    def handle_mouse(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos

            for item in self.ui_items:
                if "rect" in item and item["rect"].collidepoint(mx, my):
                    item["value"] = not item["value"]

    def text(self, text, x, y, color=COLOR_TEXT):
        surf = self.font.render(str(text), True, color)
        self.screen.blit(surf, (x, y))