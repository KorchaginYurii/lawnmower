from collections import deque


class FailureRecoveryManager:
    def __init__(self):
        self.pos_history = deque(maxlen=12)
        self.replan_history = deque(maxlen=12)

        self.stuck_counter = 0
        self.no_path_counter = 0
        self.blocked_counter = 0

        self.recovery_mode = None
        self.recovery_counts = {
            "WAIT": 0,
            "BACK_OFF": 0,
            "EXPLORE_ALT": 0,
        }

    def reset(self):
        self.pos_history.clear()
        self.replan_history.clear()

        self.stuck_counter = 0
        self.no_path_counter = 0
        self.blocked_counter = 0

        self.recovery_mode = None
        self.recovery_counts = {
            "WAIT": 0,
            "BACK_OFF": 0,
            "EXPLORE_ALT": 0,
        }

    def update(self, env, debug=None):
        self.pos_history.append(env.pos)

        if debug is not None:
            self.replan_history.append(
                1 if debug.get("need_replan", False) else 0
            )

    def detect_stuck(self):
        if len(self.pos_history) < self.pos_history.maxlen:
            return False

        unique_positions = len(set(self.pos_history))

        return unique_positions <= 3

    def detect_replan_loop(self):
        if len(self.replan_history) < self.replan_history.maxlen:
            return False

        return sum(self.replan_history) >= 11

    def report_no_path(self):
        self.no_path_counter += 1

    def report_blocked(self):
        self.blocked_counter += 1

    def clear_soft_failures(self):
        self.no_path_counter = 0
        self.blocked_counter = 0

    def choose_recovery_mode(self):
        if self.no_path_counter >= 3:
            self.recovery_mode = "EXPLORE_ALT"
            self.recovery_counts["EXPLORE_ALT"] += 1
            return self.recovery_mode

        if self.blocked_counter >= 3:
            self.recovery_mode = "BACK_OFF"
            self.recovery_counts["BACK_OFF"] += 1
            return self.recovery_mode

        if self.detect_stuck():
            self.recovery_mode = "BACK_OFF"
            self.recovery_counts["BACK_OFF"] += 1
            return self.recovery_mode

        #if self.detect_replan_loop():
        #    self.recovery_mode = "WAIT"
        #    self.recovery_counts["WAIT"] += 1
        #    return self.recovery_mode

        self.recovery_mode = None
        return None