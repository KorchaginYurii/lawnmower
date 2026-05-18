from core.global_planner import AStarPlanner
from env.cabbage_env import CabbageEnv
from core.config import MAP_H, MAP_W

if __name__ == "__main__":
    env = CabbageEnv(MAP_H, MAP_W)
    env.reset()

    planner = AStarPlanner()

    start = env.pos
    goal = env.start_pos

    path = planner.find_path(env, start, goal)

    print("start:", start)
    print("goal:", goal)
    print("path:", path)
    print("path length:", len(path) if path else None)