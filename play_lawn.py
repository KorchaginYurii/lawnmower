import time
import pygame

from env.lawn_env import LawnEnv
from adapters.lawn_hybrid_adapter import LawnHybridAdapter
from agents.lawn_sweep_agent import LawnSweepAgent
from ui.lawn_pygame_renderer import LawnRenderer


def main():
    lawn = LawnEnv(
        width_m=42,
        height_m=45,
        cell_size_m=0.25,
        robot_size_m=0.5,
        max_energy=100.0,
    )

    lawn.reset_realistic_lawn(
        object_count=10,
        seed=101,
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