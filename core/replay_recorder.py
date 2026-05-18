import pickle
import os
from datetime import datetime

class ReplayRecorder:
    def __init__(self):
        self.frames = []

    def reset(self):
        self.frames = []

    def record(self, env, debug=None):
        frame = {
            "pos": env.pos,
            "heading": env.heading,
            "energy": env.energy_system.energy,

            "grid": env.grid.copy(),

            "visit_count": getattr(env, "visit_count", None),
            "turn_count": getattr(env, "turn_count", None),

            "mode": debug.get("mode") if debug else None,
            "goal": debug.get("goal") if debug else None,
            "path": debug.get("path") if debug else None,

            "frontiers": debug.get("frontiers") if debug else None,
            "frontier_target": debug.get("frontier_target") if debug else None,

            "sector": debug.get("sector") if debug else None,

            "memory_map": debug.get("memory_map") if debug else None,
            "memory_seen": debug.get("memory_seen") if debug else None,
            "start_pos": env.start_pos,
            "initial_grid": env.initial_grid.copy(),
            "step": env.steps,
            "max_steps": env.max_steps,

            "max_energy": env.energy_system.max_energy,
            "knife_on": env.knife_on,

            "sector_h": debug.get("sector_h") if debug else 5,
            "sector_w": debug.get("sector_w") if debug else 5,

            "required_energy": debug.get("required_energy") if debug else 0.0,
            "energy_margin": debug.get("energy_margin") if debug else 0.0,

            "energy_per_cabbage": debug.get("energy_per_cabbage") if debug else 0.0,
            "total_turns": debug.get("total_turns") if debug else 0,
            "overlap_rate": debug.get("overlap_rate") if debug else 0.0,
            "sector_switches": debug.get("sector_switches") if debug else 0,

            "dynamic_obstacles": (
                list(env.dynamic_obstacles.positions())
                if hasattr(env, "dynamic_obstacles")
                else []
            ),
            "dynamic_predictions": (
                dict(env.dynamic_obstacles.predicted_positions())
                if hasattr(env, "dynamic_obstacles")
                else {}
            ),
        }

        self.frames.append(frame)




    def save(self, folder="replays", name=None):
        os.makedirs(folder, exist_ok=True)

        if name is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"replay_{stamp}.pkl"

        path = os.path.join(folder, name)

        with open(path, "wb") as f:
            pickle.dump(self.frames, f)

        print(f"✅ Replay saved: {path}")
        return path