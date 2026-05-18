import pickle
import time
import pygame
import numpy as np
import os
from ui.pygame_renderer import Renderer


class ReplayEnvView:
    """
    Лёгкая оболочка, чтобы renderer видел replay-кадр как env.
    """
    def __init__(self, frame):
        self.grid = frame["grid"]
        self.pos = tuple(frame["pos"])
        self.heading = frame.get("heading", 0)

        self.start_pos = tuple(frame.get("start_pos", (0, 0)))

        self.visit_count = frame.get("visit_count", np.zeros_like(self.grid, dtype=np.float32))
        self.turn_count = frame.get("turn_count", np.zeros_like(self.grid, dtype=np.float32))

        self.steps = frame.get("step", 0)
        self.max_steps = frame.get("max_steps", 0)

        self.initial_grid = frame.get("initial_grid", self.grid.copy())

        self.energy_system = type("EnergyView", (), {})()
        self.energy_system.energy = frame.get("energy", 0.0)
        self.energy_system.max_energy = frame.get("max_energy", 100.0)

        dynamic_positions = frame.get("dynamic_obstacles", [])

        self.dynamic_obstacles = type("DynamicObstacleView", (), {})()
        self.dynamic_obstacles.positions = lambda: set(map(tuple, dynamic_positions))

def load_replay(path="replays/last_replay.pkl"):
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


def main():
    path = choose_replay()

    if path is None:
        return

    frames = load_replay(path)
    print("Loaded:", path)

    if not frames:
        print("Replay is empty")
        return
    renderer = Renderer()

    idx = 0
    paused = False
    speed = 0.08

    print("Controls:")
    print("SPACE - pause/play")
    print("LEFT/RIGHT - step frame")
    print("UP/DOWN - speed")
    print("ESC - quit")

    running = True

    while running:
        for event in pygame.event.get():
            renderer.handle_mouse(event)

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_RIGHT:
                    idx = min(len(frames) - 1, idx + 1)

                elif event.key == pygame.K_LEFT:
                    idx = max(0, idx - 1)

                elif event.key == pygame.K_UP:
                    speed = max(0.01, speed * 0.7)

                elif event.key == pygame.K_DOWN:
                    speed = min(1.0, speed * 1.3)

        frame = frames[idx]
        env_view = ReplayEnvView(frame)
        debug = make_debug(frame)

        debug["step"] = idx
        debug["max_steps"] = len(frames)

        renderer.draw(env_view, debug)

        if not paused:
            idx += 1
            if idx >= len(frames):
                idx = len(frames) - 1
                paused = True

        time.sleep(speed)

    pygame.quit()

def choose_replay(folder="replays"):
    files = [
        f for f in os.listdir(folder)
        if f.endswith(".pkl")
    ]

    files.sort(reverse=True)

    if not files:
        print("No replay files found")
        return None

    print("\nAvailable replays:")
    for i, f in enumerate(files):
        print(f"{i}: {f}")

    idx = int(input("\nChoose replay number: "))
    return os.path.join(folder, files[idx])

if __name__ == "__main__":
    main()