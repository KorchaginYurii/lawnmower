import time
import pygame

from env.lawn_env import LawnEnv
from adapters.lawn_hybrid_adapter import LawnHybridAdapter
from agents.lawn_sweep_agent import LawnSweepAgent
from ui.lawn_renderer import LawnRenderer
from core.config import (
    LAWN_PRESET,
    LAWN_PRESETS,
    CELL_SIZE_M,
    ROBOT_SIZE_M,
    LAWNMOWER_MAX_ENERGY,
)
import inspect
import agents.lawn_sweep_agent as lawn_agent_mod
from core.tuning_config import runtime_config



def __repr__(self):
    return str(self.values)
def main():
    runtime_config.reset()

    runtime_config.load_profile(
        "configs.tuned_stable"
    )
    runtime_config.set(
        "USE_ADAPTIVE_TRAFFIC",
        False,
    )


    print("\n========== ACTIVE PROFILE ==========")
    print(
        runtime_config.get(
            "PROFILE_NAME",
            "unknown"
        )
    )
    print("====================================")

    preset = LAWN_PRESETS[LAWN_PRESET]

    lawn = LawnEnv(
        width_m=preset["width_m"],
        height_m=preset["height_m"],

        cell_size_m=CELL_SIZE_M,
        robot_size_m=ROBOT_SIZE_M,

        max_energy=LAWNMOWER_MAX_ENERGY,
    )

    lawn.reset_realistic_lawn(
        object_count=preset["object_count"],
        seed=103,
        border_margin=preset["border_margin"],
        max_object_size=preset["max_object_size"],
    )

    adapter = LawnHybridAdapter(lawn)

    agent = LawnSweepAgent()
    agent.reset()

    renderer = LawnRenderer(
        map_shape=lawn.grid.shape,
    )

    paused = False
    delay = 0.03

    running = True

    while running:
        for event in pygame.event.get():
            renderer.handle_mouse(event)

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_UP:
                    delay = max(0.005, delay * 0.7)

                elif event.key == pygame.K_DOWN:
                    delay = min(0.5, delay * 1.3)

                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not paused:
            adapter.sync_from_env()

            action, debug = agent.act(adapter)
            reward, done = adapter.step(action)

            debug["reward"] = reward
            debug["knife_on_real"] = lawn.knife_on
            debug["cut_cells_last"] = getattr(adapter, "last_cut_cells", 0)
            debug["cell_under_robot"] = lawn.grid[lawn.pos]
            debug["knife_on_real"] = getattr(adapter, "last_knife_on", lawn.knife_on)

            renderer.draw(lawn, debug)

            if done:
                paused = True
                print("DONE")

        else:
            renderer.draw(lawn, {})

        time.sleep(delay)

    pygame.quit()


if __name__ == "__main__":
    main()