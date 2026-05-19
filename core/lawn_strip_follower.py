from core.config import ACTIONS, WAIT_ACTION
from core.lawn_lane_memory import LawnLaneMemory


class LawnStripFollower:
    """
    Lane-aware strip follower.

    Задачи:
    - ехать по текущей полосе
    - помнить завершенные полосы
    - не начинать каждый раз заново после recharge
    - выбирать следующую незавершенную полосу
    """

    def __init__(self):
        self.direction = 0       # RIGHT
        self.lane_shift_dir = 2  # DOWN
        self.mode = "FORWARD"

        self.memory = LawnLaneMemory()

        self.target_lane = None
        self.target_cell = None

    def reset(self):
        self.direction = 0
        self.lane_shift_dir = 2
        self.mode = "FORWARD"

        self.memory.reset()

        self.target_lane = None
        self.target_cell = None

    def act(self, env):
        """
        env = LawnHybridAdapter
        """

        self.memory.update(env)

        current_lane = self.memory.active_lane

        # =====================================================
        # 1. Если текущая полоса завершена — выбрать новую
        # =====================================================
        if (
            current_lane is not None
            and self.memory.is_lane_completed(env, current_lane)
        ):
            self.target_lane = self.memory.next_unfinished_lane(
                env,
                from_lane=current_lane,
            )

            if self.target_lane is not None:
                self.target_cell = self.memory.first_uncut_cell_in_lane(
                    env,
                    self.target_lane,
                    prefer_pos=env.pos,
                )

                self.mode = "GO_TO_NEXT_LANE"

                return self.move_towards_target_cell(env)

        # =====================================================
        # 2. Если есть target_cell — идем к ней
        # =====================================================
        if self.target_cell is not None:
            if env.pos == self.target_cell:
                self.target_cell = None
                self.mode = "FORWARD"
            else:
                self.mode = "GO_TO_TARGET_CELL"
                return self.move_towards_target_cell(env)

        # =====================================================
        # 3. Едем по полосе
        # =====================================================
        if self.can_move(env, self.direction):
            self.mode = "FORWARD"
            return self.direction

        # =====================================================
        # 4. Конец полосы: пробуем shift на следующую линию
        # =====================================================
        if self.can_move(env, self.lane_shift_dir):
            self.mode = "SHIFT_LANE"
            return self.lane_shift_dir

        # =====================================================
        # 5. Разворот
        # =====================================================
        self.reverse_direction()

        if self.can_move(env, self.direction):
            self.mode = "REVERSE_FORWARD"
            return self.direction

        # =====================================================
        # 6. Если всё плохо — ищем новую незавершенную полосу
        # =====================================================
        self.target_lane = self.memory.next_unfinished_lane(
            env,
            from_lane=current_lane,
        )

        if self.target_lane is not None:
            self.target_cell = self.memory.first_uncut_cell_in_lane(
                env,
                self.target_lane,
                prefer_pos=env.pos,
            )

            self.mode = "SEARCH_UNFINISHED_LANE"

            return self.move_towards_target_cell(env)

        self.mode = "ESCAPE"
        return self.safe_any_action(env)

    # =====================================================
    # TARGET MOVEMENT
    # =====================================================

    def move_towards_target_cell(self, env):
        """
        Простое движение к target_cell без A*.
        Пока избегает препятствий локально.
        """

        if self.target_cell is None:
            return self.safe_any_action(env)

        x, y = env.pos
        tx, ty = self.target_cell

        candidates = []

        for a, (dx, dy) in enumerate(ACTIONS[:4]):
            nx = x + dx
            ny = y + dy

            if not self.can_move_to(env, nx, ny):
                continue

            dist = abs(nx - tx) + abs(ny - ty)

            # штраф за движение назад по последней полосе не добавляем пока
            candidates.append((dist, a))

        if not candidates:
            return self.safe_any_action(env)

        candidates.sort(key=lambda z: z[0])
        return candidates[0][1]

    # =====================================================
    # MOVEMENT RULES
    # =====================================================

    def can_move(self, env, action):
        x, y = env.pos
        dx, dy = ACTIONS[action]

        return self.can_move_to(env, x + dx, y + dy)

    def can_move_to(self, env, nx, ny):
        if nx < 0 or ny < 0:
            return False

        if nx >= env.grid.shape[0] or ny >= env.grid.shape[1]:
            return False

        if (nx, ny) in getattr(env, "obstacles", set()):
            return False

        cell = env.grid[nx, ny]

        # adapter grid:
        # 1 = uncut grass
        # 2 = cut grass
        # -1 = obstacle/buffer
        # 0 = empty, не газон
        if cell not in (1, 2):
            return False

        return True

    def reverse_direction(self):
        if self.direction == 0:
            self.direction = 1
        elif self.direction == 1:
            self.direction = 0
        elif self.direction == 2:
            self.direction = 3
        elif self.direction == 3:
            self.direction = 2

    def safe_any_action(self, env):
        for a in range(4):
            if self.can_move(env, a):
                return a

        return WAIT_ACTION