from core.config import ACTIONS


class LawnStripFollower:
    """
    Локальный контроллер полосы.

    Идея:
    - ехать прямо по полосе
    - не вызывать A* на каждую соседнюю клетку
    - разворачиваться только когда полоса закончилась
    """

    def __init__(self):
        self.direction = 0      # 0=RIGHT, 1=LEFT, 2=DOWN, 3=UP
        self.lane_shift_dir = 2 # DOWN
        self.mode = "FORWARD"

    def reset(self):
        self.direction = 0
        self.lane_shift_dir = 2
        self.mode = "FORWARD"

    def act(self, env):
        """
        Возвращает action.
        env = LawnHybridAdapter
        """

        # 1. пробуем ехать по текущей полосе
        if self.can_move(env, self.direction):
            self.mode = "FORWARD"
            return self.direction

        # 2. если уперлись — пробуем перейти на следующую полосу
        if self.can_move(env, self.lane_shift_dir):
            self.mode = "SHIFT_LANE"
            return self.lane_shift_dir

        # 3. после сдвига меняем направление туда-обратно
        self.reverse_direction()

        if self.can_move(env, self.direction):
            self.mode = "REVERSE_FORWARD"
            return self.direction

        # 4. если всё плохо — ищем любой безопасный ход
        self.mode = "ESCAPE"
        return self.safe_any_action(env)

    def can_move(self, env, action):
        x, y = env.pos
        dx, dy = ACTIONS[action]

        nx = x + dx
        ny = y + dy

        p = (nx, ny)

        if p in getattr(env, "obstacles", set()):
            return False

        if nx < 0 or ny < 0:
            return False

        if nx >= env.grid.shape[0] or ny >= env.grid.shape[1]:
            return False

        cell = env.grid[nx, ny]

        # можно ехать только по mowable area
        if cell not in (1, 2):
            return False

        return True

    def reverse_direction(self):
        if self.direction == 0:      # RIGHT
            self.direction = 1       # LEFT
        elif self.direction == 1:    # LEFT
            self.direction = 0       # RIGHT
        elif self.direction == 2:    # DOWN
            self.direction = 3       # UP
        elif self.direction == 3:    # UP
            self.direction = 2       # DOWN

    def safe_any_action(self, env):
        for a in range(4):
            if self.can_move(env, a):
                return a
        return 4  # WAIT