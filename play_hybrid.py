import time
import pygame

from env.cabbage_env import CabbageEnv
from agents.cabbage_agent import CabbageAgent
from agents.hybrid_agent import HybridAgent
from ui.pygame_renderer import Renderer

from core.checkpoint import CheckpointManager
from core.replay_recorder import ReplayRecorder
from core.config import MAP_H, MAP_W
from core.config import USE_LOCAL_RL

recorder = ReplayRecorder()
env = CabbageEnv(MAP_H, MAP_W)
env.reset()

if USE_LOCAL_RL:
    local_agent = CabbageAgent()
    ckpt = CheckpointManager(
        k_best=3,
        project_name="Cab4"
    )
    ckpt.load_checkpoint(local_agent)
    start_ep, best = ckpt.load_checkpoint(local_agent)


agent = HybridAgent(
    local_agent=local_agent if USE_LOCAL_RL else None,
    robot_id="robot_1"
)
agent.reset()

renderer = Renderer()

while True:
    for event in pygame.event.get():
        renderer.handle_mouse(event)

        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    action, debug = agent.act(env)
    reward, done = env.step(action)
    recorder.record(env, debug)
    renderer.draw(env, debug)

    time.sleep(0.1)

    if done:
        recorder.save()
        print("DONE")
        break