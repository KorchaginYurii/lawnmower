from env.lawn_env import LawnEnv
from adapters.lawn_hybrid_adapter import LawnHybridAdapter
from agents.hybrid_agent import HybridAgent


def run_lawn_benchmark(max_steps=5000):
    lawn = LawnEnv(
        width_m=42,
        height_m=45,
        cell_size_m=0.25,
        robot_size_m=0.5,
        max_energy=5000.0,
    )

    lawn.reset_realistic_lawn(object_count=10, seed=42)

    agent_env = LawnHybridAdapter(lawn)

    agent = HybridAgent()
    agent.reset()

    total_reward = 0.0
    last_debug = {}

    for step in range(max_steps):
        agent_env.sync_from_env()

        action, debug = agent.act(agent_env)

        reward, done = agent_env.step(action)
        total_reward += reward
        last_debug = debug

        if step % 100 == 0:
            print(
                step,
                "coverage=", round(lawn.coverage_rate(), 3),
                "overlap=", round(lawn.overlap_rate(), 3),
                "energy=", round(lawn.energy_used, 2),
                "mode=", debug.get("mode"),
                "goal=", debug.get("goal"),
            )

        if done:
            break

    result = {
        "steps": step + 1,
        "total_reward": total_reward,
        "coverage_rate": lawn.coverage_rate(),
        "overlap_rate": lawn.overlap_rate(),
        "energy_used": lawn.energy_used,
        "remaining_grass": lawn.remaining_grass(),
        "total_turns": lawn.total_turns,
        "last_mode": last_debug.get("mode"),
    }

    print("\n===== LAWN BENCHMARK RESULT =====")
    for k, v in result.items():
        print(k, "=", v)

    return result


if __name__ == "__main__":
    run_lawn_benchmark()