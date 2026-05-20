from core.config import ACTIONS


class LawnEnergyManager:
    """
    Управление энергией газонокосилки.

    Логика:
    - считаем примерную цену возврата домой
    - если энергии мало, переводим агента в RETURN_HOME
    - на базе выполняем recharge
    - после зарядки возвращаемся в SWEEP
    """

    def __init__(
        self,
        reserve=30.0,
        move_cost=0.05,
        turn_cost=0.02,
        cut_cost=0.40,
    ):
        self.reserve = reserve
        self.move_cost = move_cost
        self.turn_cost = turn_cost
        self.cut_cost = cut_cost

        self.mode = "SWEEP"

    def reset(self):
        self.mode = "SWEEP"

    def should_return_home(self, env):
        """
        env = LawnHybridAdapter
        """

        if env.pos == env.start_pos:
            return False

        energy_ratio = env.energy_system.energy / max(
            1e-9,
            env.energy_system.max_energy
        )

        if energy_ratio <= 0.30:
            return True

        return_cost = self.estimate_return_cost(env)

        return env.energy_system.energy <= return_cost + self.reserve

    def estimate_return_cost(self, env):
        """
        Быстрая оценка без A*:
        Manhattan distance * move_cost + примерный запас на повороты.
        """

        x, y = env.pos
        sx, sy = env.start_pos

        dist = abs(x - sx) + abs(y - sy)

        move_energy = dist * self.move_cost
        turn_energy = 4 * self.turn_cost

        return move_energy + turn_energy

    def estimate_path_cost(self, env, path):
        if path is None or len(path) < 2:
            return 0.0

        heading = env.heading
        cost = 0.0

        for i in range(len(path) - 1):
            x, y = path[i]
            nx, ny = path[i + 1]

            dx = nx - x
            dy = ny - y

            cost += self.move_cost

            target_heading = heading

            for h, (hx, hy) in enumerate([
                (-1, 0),
                (0, 1),
                (1, 0),
                (0, -1),
            ]):
                if (dx, dy) == (hx, hy):
                    target_heading = h
                    break

            diff = abs(target_heading - heading)
            diff = min(diff, 4 - diff)

            cost += diff * self.turn_cost
            heading = target_heading

        return cost

    def recharge_if_home(self, env):
        """
        Возвращает True, если зарядились.
        """

        if env.pos != env.start_pos:
            return False

        if hasattr(env, "energy_system"):
            env.energy_system.recharge()
            self.mode = "SWEEP"
            return True

        return False