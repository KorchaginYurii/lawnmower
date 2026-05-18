import os
import pickle
import time
import pygame
import numpy as np

from ui.pygame_renderer import Renderer


class ReplayEnvView:
    def __init__(self, frame):
        self.grid = frame["grid"]
        self.pos = tuple(frame["pos"])
        self.heading = frame.get("heading", 0)
        self.start_pos = tuple(frame.get("start_pos", (0, 0)))

        self.visit_count = frame.get(
            "visit_count",
            np.zeros_like(self.grid, dtype=np.float32)
        )
        self.turn_count = frame.get(
            "turn_count",
            np.zeros_like(self.grid, dtype=np.float32)
        )

        self.steps = frame.get("step", 0)
        self.max_steps = frame.get("max_steps", 0)
        self.initial_grid = frame.get("initial_grid", self.grid.copy())

        self.energy_system = type("EnergyView", (), {})()
        self.energy_system.energy = frame.get("energy", 0.0)
        self.energy_system.max_energy = frame.get("max_energy", 100.0)

        dynamic_positions = frame.get("dynamic_obstacles", [])

        self.dynamic_obstacles = type("DynamicObstacleView", (), {})()
        self.dynamic_obstacles.positions = lambda: set(map(tuple, dynamic_positions))

def choose_replay(folder="replays"):
    files = [f for f in os.listdir(folder) if f.endswith(".pkl")]
    files.sort(reverse=True)

    if not files:
        print("No replay files found")
        return None

    print("\nAvailable replays:")
    for i, f in enumerate(files):
        print(f"{i}: {f}")

    idx = int(input("\nChoose replay number: "))
    return os.path.join(folder, files[idx])


def load_replay(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_debug(frame):
    return {
        "mode": frame.get("mode"),
        "goal": frame.get("goal"),
        "path": frame.get("path"),

        "frontiers": frame.get("frontiers") or [],
        "frontier_target": frame.get("frontier_target"),# or [],
        "frontier_clusters": frame.get("frontier_clusters") or [],

        "sector": frame.get("sector"),
        "sector_h": frame.get("sector_h", 5),
        "sector_w": frame.get("sector_w", 5),

        "memory_map": frame.get("memory_map"),
        "memory_seen": frame.get("memory_seen"),

        "energy": frame.get("energy", 0.0),
        "max_energy": frame.get("max_energy", 100.0),

        "heading": frame.get("heading", 0),
        "knife_on": frame.get("knife_on", False),

        "reward": frame.get("reward", 0.0),
        "total_reward": frame.get("total_reward", 0.0),

        "energy_per_cabbage": frame.get("energy_per_cabbage", 0.0),
        "total_turns": frame.get("total_turns", 0),
        "overlap_rate": frame.get("overlap_rate", 0.0),
        "sector_switches": frame.get("sector_switches", 0),

        "required_energy": frame.get("required_energy", 0.0),
        "energy_margin": frame.get("energy_margin", 0.0),
    }


def mission_summary(frames):
    last = frames[-1]

    grid = last["grid"]
    initial = last.get("initial_grid", grid)

    total = int(np.sum(initial == 1))
    remaining = int(np.sum(grid == 1))
    collected = total - remaining

    energy_start = frames[0].get("energy", 100.0)
    energy_end = last.get("energy", 0.0)
    energy_used = energy_start - energy_end

    return {
        "frames": len(frames),
        "collected": collected,
        "total": total,
        "energy_used": energy_used,
        "turns": last.get("total_turns", 0),
        "overlap": last.get("overlap_rate", 0.0),
        "mode": last.get("mode"),
    }


def print_summary(name, frames):
    s = mission_summary(frames)

    print(f"\n{name}")
    print("-" * 40)
    print(f"frames:       {s['frames']}")
    print(f"cabbage:      {s['collected']}/{s['total']}")
    print(f"energy used:  {s['energy_used']:.2f}%")
    print(f"turns:        {s['turns']}")
    print(f"overlap:      {s['overlap']:.2f}")
    print(f"final mode:   {s['mode']}")


def main():
    print("Choose LEFT replay")
    left_path = choose_replay()

    print("\nChoose RIGHT replay")
    right_path = choose_replay()

    if left_path is None or right_path is None:
        return

    left_frames = load_replay(left_path)
    right_frames = load_replay(right_path)

    print_summary("LEFT", left_frames)
    print_summary("RIGHT", right_frames)

    left_renderer = Renderer()
    right_renderer = Renderer()

    cell_area_w = left_renderer.screen.get_width()
    cell_area_h = left_renderer.screen.get_height()

    screen = pygame.display.set_mode((cell_area_w * 2, cell_area_h))
    pygame.display.set_caption("Replay Compare")

    idx = 0
    paused = False
    speed = 0.08

    running = True

    print("\nControls:")
    print("SPACE - pause/play")
    print("LEFT/RIGHT - step")
    print("UP/DOWN - speed")
    print("ESC - quit")

    while running:
        for event in pygame.event.get():
            left_renderer.handle_mouse(event)
            right_renderer.handle_mouse(event)

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    idx += 1
                elif event.key == pygame.K_LEFT:
                    idx -= 1
                elif event.key == pygame.K_UP:
                    speed = max(0.01, speed * 0.7)
                elif event.key == pygame.K_DOWN:
                    speed = min(1.0, speed * 1.3)

        idx = max(0, idx)

        li = min(idx, len(left_frames) - 1)
        ri = min(idx, len(right_frames) - 1)

        left_env = ReplayEnvView(left_frames[li])
        right_env = ReplayEnvView(right_frames[ri])

        left_debug = make_debug(left_frames[li])
        right_debug = make_debug(right_frames[ri])

        left_debug["step"] = li
        left_debug["max_steps"] = len(left_frames)

        right_debug["step"] = ri
        right_debug["max_steps"] = len(right_frames)

        # рисуем в собственные surfaces renderer-ов
        left_renderer.draw(left_env, left_debug)
        right_renderer.draw(right_env, right_debug)

        # копируем их экраны в общий экран
        screen.blit(left_renderer.screen, (0, 0))
        screen.blit(right_renderer.screen, (cell_area_w, 0))

        # подписи
        font = pygame.font.SysFont("consolas", 18)

        left_label = font.render("LEFT", True, (255, 255, 255))
        right_label = font.render("RIGHT", True, (255, 255, 255))

        screen.blit(left_label, (10, 5))
        screen.blit(right_label, (cell_area_w + 10, 5))

        pygame.display.flip()

        if not paused:
            idx += 1

            if idx >= max(len(left_frames), len(right_frames)):
                idx = max(len(left_frames), len(right_frames)) - 1
                paused = True

        time.sleep(speed)

    pygame.quit()


if __name__ == "__main__":
    main()