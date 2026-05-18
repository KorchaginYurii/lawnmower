import pygame
import numpy as np
from core.config import MAP_H, MAP_W, WINDOW_W, WINDOW_H, HUD_W, HUD_H, MIN_CELL,MAX_CELL
#from agents.cabbage_agent import CabbageAgent
#agent = CabbageAgent()
import torch


MARGIN = 2

WHITE = (240,240,240)
GREEN = (80,200,80)
BLACK = (30,30,30)
RED = (220,50,50)
BLUE = (80,80,255)


class Renderer:
    def __init__(self):
        pygame.init()

        usable_w = WINDOW_W - HUD_W
        usable_h = WINDOW_H - HUD_H

        cell_w = usable_w // MAP_W
        cell_h = usable_h // MAP_H

        self.cell = max(
            MIN_CELL,
            min(MAX_CELL, min(cell_w, cell_h))
        )


        self.map_h = MAP_H * self.cell
        self.map_w = MAP_W * self.cell

        self.screen = pygame.display.set_mode(
            (self.map_w + HUD_W, self.map_h + HUD_H)
        )

        self.overlay = pygame.Surface(
            (self.map_w, self.map_h),
            pygame.SRCALPHA
        )

        self.show_path = True
        self.show_goal = True
        self.show_sector = True
        self.show_visited = False
        self.show_coverage = False
        self.show_turns = False
        self.font = pygame.font.SysFont("consolas", 18)
        self.show_memory_map = False

        self.ui_items = [
            {"name": "A* Path", "key": "show_path", "value": True},
            {"name": "Coverage Heatmap", "key": "show_coverage", "value": False},
            {"name": "Visited Heatmap", "key": "show_visited", "value": False},
            {"name": "Turn Heatmap", "key": "show_turns", "value": False},
            {"name": "Sector", "key": "show_sector", "value": True},
            {"name": "Goal", "key": "show_goal", "value": True},
            {"name": "Memory Map", "key": "show_memory_map", "value": False},
        ]

    def draw(self, env, debug=None, agent=None):

        self.screen.fill((20, 20, 20))

        # ===== UI =====
        y_end = self.draw_ui()
        self.draw_legend(y_end)

        for item in self.ui_items:
            setattr(self, item["key"], item["value"])

        # =====================================================
        # MAP
        # =====================================================
        memory_map = None
        memory_seen = None

        if debug is not None and self.show_memory_map:
            memory_map = debug.get("memory_map", None)
            memory_seen = debug.get("memory_seen", None)

        for i in range(env.grid.shape[0]):
            for j in range(env.grid.shape[1]):

                rect = (
                    j * self.cell,
                    i * self.cell,
                    self.cell - MARGIN,
                    self.cell - MARGIN
                )

                if memory_map is not None:
                    val = memory_map[i][j]
                    seen = memory_seen[i][j] if memory_seen is not None else 1

                    if seen == 0:
                        color = (70, 70, 70)  # unknown
                    elif val == 1:
                        color = GREEN  # known cabbage
                    elif val == -1:
                        color = BLACK  # known obstacle
                    elif val == 2:
                        color = (0, 0, 255)  # base
                    else:
                        color = WHITE  # known empty

                else:
                    val = env.grid[i][j]

                    if val == 1:
                        color = GREEN
                    elif val == -1:
                        color = BLACK
                    else:
                        color = WHITE

                if hasattr(env, "dynamic_obstacles"):
                    for x, y in env.dynamic_obstacles.positions():
                        pygame.draw.circle(
                            self.screen,
                       (255, 120, 0),
                      (y * self.cell + self.cell // 2, x * self.cell + self.cell // 2),
                            self.cell // 3
                        )


                pygame.draw.rect(self.screen, color, rect)
        # =====================================================
        # HEATMAPS
        # =====================================================
        if self.show_visited and hasattr(env, "visited"):
            self.draw_visited_heatmap(env.visited)

        if self.show_coverage and hasattr(env, "visit_count"):
            self.draw_coverage_heatmap(env.visit_count)

        if self.show_turns and hasattr(env, "turn_count"):
            self.draw_turn_heatmap(env.turn_count)

        # ===========================================
        # FRONTIER
        # ===========================================
        if debug is not None and "frontiers" in debug:
            for fx, fy in debug["frontiers"]:
                pygame.draw.circle(
                    self.screen,
                    (180, 0, 255),
                    (fy * self.cell + self.cell // 2, fx * self.cell + self.cell // 2),
                    3
                )
        # =====================================================
        # FRONTIER CLUSTERS
        # =====================================================
        frontier_clusters = []

        if debug is not None:
            frontier_clusters = debug.get("frontier_clusters") or []

        for cluster in frontier_clusters:
            for fx, fy in cluster:
                pygame.draw.circle(
                    self.screen,
                    (160, 0, 220),
                    (fy * self.cell + self.cell // 2, fx * self.cell + self.cell // 2),
                    3
                )
        # =====================================================
        # SELECTED FRONTIER
        # =====================================================
        if debug is not None and debug.get("frontier_target", None) is not None:
            fx, fy = debug["frontier_target"]

            pygame.draw.circle(
                self.screen,
                (255, 0, 255),
                (fy * self.cell + self.cell // 2, fx * self.cell + self.cell // 2),
                9,
                3
            )
        # =====================================================
        # SECTOR
        # =====================================================
        if (
                self.show_sector and
                debug is not None and
                "sector" in debug and
                debug["sector"] is not None
        ):
            sx, sy = debug["sector"]

            sector_h = debug.get("sector_h", 5)
            sector_w = debug.get("sector_w", 5)

            x1 = sx * sector_h
            y1 = sy * sector_w

            w = sector_w * self.cell
            h = sector_h * self.cell

            pygame.draw.rect(
                self.screen,
                (255, 255, 0),
                (y1 * self.cell, x1 * self.cell, w, h),
                3
            )

        # =====================================================
        # GOAL
        # =====================================================
        if (
                self.show_goal and
                debug is not None and
                "goal" in debug and
                debug["goal"] is not None
        ):
            gx, gy = debug["goal"]

            pygame.draw.rect(
                self.screen,
                (255, 200, 0),
                (
                    gy * self.cell + 8,
                    gx * self.cell + 8,
                    self.cell - 16,
                    self.cell - 16
                ),
                3
            )

        # =====================================================
        # PATH
        # =====================================================
        if (
                self.show_path and
                debug is not None and
                "path" in debug and
                debug["path"] is not None
        ):

            path = debug["path"]

            for i, (px, py) in enumerate(path):

                radius = 4

                if i == 0:
                    color = (0, 255, 255)

                elif i == len(path) - 1:
                    color = (255, 255, 0)
                    radius = 6

                else:
                    color = (0, 180, 255)

                pygame.draw.circle(
                    self.screen,
                    color,
                    (
                        py * self.cell + self.cell // 2,
                        px * self.cell + self.cell // 2
                    ),
                    radius
                )

                # соединяющие линии
                if i > 0:
                    prev_x, prev_y = path[i - 1]

                    pygame.draw.line(
                        self.screen,
                        (0, 140, 220),
                        (
                            prev_y * self.cell + self.cell // 2,
                            prev_x * self.cell + self.cell // 2
                        ),
                        (
                            py * self.cell + self.cell // 2,
                            px * self.cell + self.cell // 2
                        ),
                        2
                    )

        # =====================================================
        # START BASE
        # =====================================================
        sx, sy = env.start_pos

        pygame.draw.rect(
            self.screen,
            (0, 0, 255),
            (
                sy * self.cell,
                sx * self.cell,
                self.cell - MARGIN,
                self.cell - MARGIN
            ),
            5
        )

        # =====================================================
        # AGENT
        # =====================================================
        x, y = env.pos

        rect = pygame.Rect(
            y * self.cell,
            x * self.cell,
            self.cell - MARGIN,
            self.cell - MARGIN
        )

        pygame.draw.rect(
            self.screen,
            RED,
            rect,
            3
        )

        for rid, pos in debug.get("robot_positions", {}).items():
            rx, ry = pos
            pygame.draw.circle(
                self.screen,
                (0, 255, 180),
                (ry * self.cell + self.cell // 2, rx * self.cell + self.cell // 2),
                10,
                2
            )

        predictions = debug.get("dynamic_predictions", {}) if debug else {}

        for (px, py), t in predictions.items():
            if 0 <= px < env.grid.shape[0] and 0 <= py < env.grid.shape[1]:
                pygame.draw.circle(
                    self.screen,
                    (255, 180, 0),
                    (
                        py * self.cell + self.cell // 2,
                        px * self.cell + self.cell // 2
                    ),
                    max(2, self.cell // 5),
                    1
                )

        if debug is not None and debug.get("dynamic_traffic") is not None:
            traffic = debug["dynamic_traffic"]
            max_v = np.max(traffic)

            if max_v > 0:
                overlay = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)

                for i in range(traffic.shape[0]):
                    for j in range(traffic.shape[1]):
                        v = traffic[i, j] / max_v

                        if v <= 0.05:
                            continue

                        alpha = int(120 * v)

                        pygame.draw.rect(
                            overlay,
                            (255, 80, 0, alpha),
                            (
                                j * self.cell,
                                i * self.cell,
                                self.cell,
                                self.cell
                            )
                        )





                self.screen.blit(overlay, (0, 0))

        # =====================================================
        # HUD
        # =====================================================
        if debug is not None:



            total = np.sum(env.initial_grid == 1)
            remaining = np.sum(env.grid == 1)
            collected = total - remaining

            step = env.steps
            max_steps = env.max_steps

            reward = debug.get("reward", 0.0)
            total_reward = debug.get("total_reward", 0.0)

            mode = debug.get("mode", "COLLECT")
            dist_home = debug.get("dist_home", 0)

            energy = debug.get("energy", None)
            max_energy = debug.get("max_energy", None)

            heading = debug.get("heading", 0)
            knife_on = debug.get("knife_on", False)

            mcts_depth = debug.get("depth", 0)


            # ===== EFFICIENCY =====
            energy_used = (
                max_energy - energy
                if energy is not None
                else 0.0
            )

            if energy_used > 0:
                efficiency = collected / energy_used
            else:
                efficiency = 0.0


            # ===== HEADING =====
            arrows = ["↑", "→", "↓", "←"]


            # ===== KNIFE =====
            knife_color = (
                (255, 80, 80)
                if knife_on
                else
                (180, 180, 180)
            )


            # ===== Dead Energy Prediction ======
            required_energy = debug.get("required_energy", None)
            energy_margin = debug.get("energy_margin", None)


            frontier_clusters = debug.get("frontier_clusters", [])
            frontier_count = sum(len(c) for c in frontier_clusters)

            risk_mode = debug.get("risk_mode", "NORMAL")

            risk_color = (255, 255, 255)

            if risk_mode == "CAREFUL":
                risk_color = (255, 180, 80)
            elif risk_mode == "SAFE_RETURN":
                risk_color = (255, 80, 80)

            recovery_mode = debug.get("recovery_mode", None)

            self.draw_compact_hud(env, debug)


        pygame.display.flip()

    def to_screen(self, x, y):
        return (y * self.cell + self.cell // 2, x * self.cell + self.cell // 2)

    def draw_visited_heatmap(self, visited):
        overlay = pygame.Surface((self.map_h, self.map_w), pygame.SRCALPHA)

        max_v = np.max(visited)

        if max_v <= 1e-6:
            return

        for i in range(visited.shape[0]):
            for j in range(visited.shape[1]):
                val = float(visited[i][j]) / max_v

                if val <= 0.01:
                    continue

                # синий -> красный
                red = int(255 * val)
                blue = int(255 * (1.0 - val))
                alpha = int(120 * val)

                color = (red, 40, blue, alpha)

                rect = (
                    j * self.cell,
                    i * self.cell,
                    self.cell - MARGIN,
                    self.cell - MARGIN
                )

                pygame.draw.rect(overlay, color, rect)

        self.screen.blit(overlay, (0, 0))

    def draw_text(self, text, x, y, color=(255, 255, 255)):
        surface = self.font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def draw_bar(self, x, y, w, h, value, color=(0, 255, 0)):
        pygame.draw.rect(self.screen, (80, 80, 80), (x, y, w, h))
        pygame.draw.rect(self.screen, color, (x, y, int(w * value), h))

    def draw_legend(self, y_start):
        x = self.map_w + 10
        y = y_start + 10  # небольшой отступ

        items = [
            ("Agent", (255, 0, 0)),
            ("Start/Base", (0, 0, 255)),
            ("Goal", (255, 200, 0)),
            ("A* Path", (0, 180, 255)),
            ("Current Sector", (255, 255, 0)),
            ("Coverage Heatmap", (255, 60, 60)),
            ("Visited Heatmap", (255, 40, 80)),
            ("Energy", (255, 255, 0)),
            ("Turn Heatmap", (255, 80, 0)),
            ("Frontier", (180, 0, 255)),
            ("Frontier Cluster", (160, 0, 220)),
            ("Selected Frontier", (255, 0, 255)),
            ("Unknown", (70, 70, 70)),
            ("Known Empty", (240, 240, 240)),
            ("Known Cabbage", (80, 200, 80)),
            ("Known Obstacle", (30, 30, 30)),
            ("Dynamic Obstacle", (255, 120, 0)),
        ]

        for name, color in items:
            pygame.draw.rect(self.screen, color, (x, y, 15, 15))
            self.draw_text(name, x + 20, y)
            y += 20

    def draw_ui(self):
        x = self.map_w + 10
        y = 20

        for item in self.ui_items:
            rect = pygame.Rect(x, y, 16, 16)
            pygame.draw.rect(self.screen, (200, 200, 200), rect, 2)

            if item["value"]:
                pygame.draw.line(self.screen, (0, 255, 0), (x, y), (x + 16, y + 16), 2)
                pygame.draw.line(self.screen, (0, 255, 0), (x + 16, y), (x, y + 16), 2)

            self.draw_text(item["name"], x + 25, y - 2)

            item["rect"] = rect
            y += 25

        return y  # 🔥 ВАЖНО

    def handle_mouse(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos

            for item in self.ui_items:
                if "rect" in item and item["rect"].collidepoint(mx, my):
                    item["value"] = not item["value"]

    def draw_energy_bar(self, x, y, w, h, energy, max_energy):
        ratio = max(0.0, min(1.0, energy / (max_energy + 1e-6)))

        pygame.draw.rect(self.screen, (80, 80, 80), (x, y, w, h))
        pygame.draw.rect(self.screen, (255, 220, 0), (x, y, int(w * ratio), h))
        pygame.draw.rect(self.screen, (220, 220, 220), (x, y, w, h), 2)

        self.draw_text(f"{energy:.1f}/{max_energy:.1f}%", x + w + 8, y - 3)

    def draw_coverage_heatmap(self, visit_count):
        overlay = pygame.Surface((self.map_h, self.map_w), pygame.SRCALPHA)

        max_v = np.max(visit_count)

        if max_v <= 0:
            return

        for i in range(visit_count.shape[0]):
            for j in range(visit_count.shape[1]):
                count = float(visit_count[i][j])

                if count <= 0:
                    continue

                val = min(1.0, count / max_v)

                # 1 посещение — мягкий голубой
                # много посещений — красный
                red = int(255 * val)
                green = int(120 * (1.0 - val))
                blue = int(255 * (1.0 - val))
                alpha = int(70 + 130 * val)

                color = (red, green, blue, alpha)

                rect = (
                    j * self.cell,
                    i * self.cell,
                    self.cell - MARGIN,
                    self.cell - MARGIN
                )

                pygame.draw.rect(overlay, color, rect)

                # если клетка посещена много раз — подпишем число
                if count >= 3:
                    self.draw_text(
                        str(int(count)),
                        j * self.cell + 5,
                        i * self.cell + 5,
                        (255, 255, 255)
                    )

        self.screen.blit(overlay, (0, 0))

    def draw_turn_heatmap(self, turn_count):
        overlay = pygame.Surface((self.map_h, self.map_w), pygame.SRCALPHA)

        max_v = np.max(turn_count)

        if max_v <= 0:
            return

        for i in range(turn_count.shape[0]):
            for j in range(turn_count.shape[1]):
                count = float(turn_count[i][j])

                if count <= 0:
                    continue

                val = min(1.0, count / max_v)

                # повороты: жёлтый -> красный
                red = 255
                green = int(220 * (1.0 - val))
                blue = 0
                alpha = int(80 + 160 * val)

                color = (red, green, blue, alpha)

                rect = (
                    j * self.cell,
                    i * self.cell,
                    self.cell - MARGIN,
                    self.cell - MARGIN
                )

                pygame.draw.rect(overlay, color, rect)

                if count >= 2:
                    self.draw_text(
                        str(int(count)),
                        j * self.cell + 5,
                        i * self.cell + 5,
                        (255, 255, 255)
                    )

        self.screen.blit(overlay, (0, 0))

    def draw_compact_hud(self, env, debug):

        y0 = self.map_h + 5
        x0 = 10

        gap = 210
        row_h = 20

        def hud_item(col, row, text, color=(255, 255, 255)):
            self.draw_text(
                text,
                x0 + col * gap,
                y0 + row * row_h,
                color
            )

        total = np.sum(env.initial_grid == 1)
        remaining = np.sum(env.grid == 1)
        collected = total - remaining

        hud_item(0, 0, f"STEP {env.steps}/{env.max_steps}")
        hud_item(1, 0, f"CAB {collected}/{total}")
        hud_item(2, 0, f"MODE {debug.get('mode', '-')}")
        hud_item(3, 0, f"RISK {debug.get('risk_mode', '-')}")
        hud_item(4, 0, f"ENERGY {debug.get('energy', 0):.1f}")

        hud_item(0, 1, f"E/CAB {debug.get('energy_per_cabbage', 0):.2f}")
        hud_item(1, 1, f"TURNS {debug.get('total_turns', 0)}")
        hud_item(2, 1, f"OVER {debug.get('overlap_rate', 0):.2f}")
        hud_item(3, 1, f"SECT_SW {debug.get('sector_switches', 0)}")
        hud_item(4, 1, f"SECTOR {debug.get('sector', '-')}")
        hud_item(5, 1, f"REPLAN {debug.get('replan_cooldown', 0)}")

        recovery = debug.get("recovery_mode", None)
        frontiers = debug.get("frontier_count", 0)

        hud_item(0, 2, f"FRONT {frontiers}")
        hud_item(1, 2, f"RECH {getattr(env, 'recharge_count', 0)}")
        hud_item(2, 2, f"REQ {debug.get('required_energy', 0):.1f}")
        hud_item(3, 2, f"MARGIN {debug.get('energy_margin', 0):.1f}")
        hud_item(4, 2, f"REC {recovery or '-'}")