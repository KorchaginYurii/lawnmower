import matplotlib.pyplot as plt
from env.lawn_env import LawnEnv


def visualize_lawn_map(seed=42, object_count=10):
    env = LawnEnv(
        width_m=42,
        height_m=45,
        cell_size_m=0.25,
        robot_size_m=0.5,
    )

    env.reset_realistic_lawn(
        object_count=object_count,
        seed=seed,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(env.grid, interpolation="nearest")
    plt.title(
        f"Lawn map | grass={env.remaining_grass()} | "
        f"size={env.grid.shape}"
    )
    plt.axis("off")
    plt.show()

    return env


if __name__ == "__main__":
    visualize_lawn_map()