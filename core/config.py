import torch

# =========================================
# CONFIG VERSION
# =========================================

CONFIG_VERSION = "knife control_v1"

# ========================================
# FLAGS
# ========================================
USE_LOCAL_RL = False
USE_HIERARCHICAL_PLANNER = False
USE_PORTAL_PLANNER = False
USE_ADAPTIVE_TRAFFIC = False
USE_HIGH_LEVEL_POLICY = True

# =========================================
# WINDOW / UI
# =========================================

WINDOW_W = 1600
WINDOW_H = 900

HUD_W = 260
HUD_H = 80

MIN_CELL = 4
MAX_CELL = 64

# =========================================
# ENV
# =========================================

MAP_H = 50
MAP_W = 100

VISION_SIZE = 7

# =========================================
# LAWNMOWER MAP PRESETS
# =========================================

LAWN_PRESET = "small"

LAWN_PRESETS = {
    "tiny": {
        "width_m": 6,
        "height_m": 5,
        "object_count": 1,
        "max_object_size": 3,
        "border_margin": 1,
    },
    "small": {
        "width_m": 12,
        "height_m": 10,
        "object_count": 3,
        "max_object_size": 5,
        "border_margin": 1,
    },
    "medium": {
        "width_m": 24,
        "height_m": 20,
        "object_count": 6,
        "max_object_size": 10,
        "border_margin": 2,
    },
    "full": {
        "width_m": 42,
        "height_m": 45,
        "object_count": 10,
        "max_object_size": 24,
        "border_margin": 4,
    },
}

CELL_SIZE_M = 0.25
ROBOT_SIZE_M = 0.50
LAWNMOWER_MAX_ENERGY = 100.0

# =========================================
# ACTIONS
# =========================================

ACTIONS = [
    (0, 1),    # RIGHT
    (0, -1),   # LEFT
    (1, 0),    # DOWN
    (-1, 0),   # UP
    (0, 0)     # WAIT
]

WAIT_ACTION = 4
NUM_ACTIONS = len(ACTIONS)

DIRECTIONS = [
    (-1, 0),   # UP
    (0, 1),    # RIGHT
    (1, 0),    # DOWN
    (0, -1),   # LEFT
]

DIR_NAMES = [
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT"
]

# =========================================
# AUTO-TUNED PARAMETERS — LAWNMOWER
# =========================================

# Coverage traffic / hot corridor avoidance
VISIT_WEIGHT = 0.04
CUT_WEIGHT = 0.08
TRAFFIC_MAX_PENALTY = 8.0

# Coverage cell ordering
CELL_DISTANCE_WEIGHT = 1.5
CELL_TRAFFIC_WEIGHT = 0.8
CELL_NEIGHBOR_BONUS = 20.0
CELL_UNCUT_WEIGHT = 3.0
CELL_RETURN_HOME_WEIGHT = 2.0

# Recovery
RECOVERY_MAX_TARGET_DIST = 30
RECOVERY_VISIT_PENALTY = 5.0
RECOVERY_CELL_SWITCH_PENALTY = 25.0
RECOVERY_CANDIDATES = 50

# Energy / return
ENERGY_RESERVE = 30.0
RETURN_HOME_RATIO = 0.30

# Watchdogs
NO_CUT_LIMIT = 20
STUCK_LIMIT = 8
LOOP_WINDOW = 20
LOOP_UNIQUE_LIMIT = 4
LOOP_TRIGGER = 3

# Benchmark early stop
MIN_TOTAL_REWARD = -5000.0
MIN_RECENT_REWARD = -1000.0
RECENT_REWARD_WINDOW = 300
MAX_RECHARGES = 30
SPIN_TURN_RATIO = 0.95
SPIN_MIN_COVERAGE = 0.20

# =========================================
# RUNTIME TUNABLES — LAWNMOWER
# =========================================

TUNABLES = {
    # Coverage traffic / hot corridor avoidance
    "VISIT_WEIGHT": VISIT_WEIGHT,
    "CUT_WEIGHT": CUT_WEIGHT,
    "TRAFFIC_MAX_PENALTY": TRAFFIC_MAX_PENALTY,

    # Coverage cell ordering
    "CELL_DISTANCE_WEIGHT": CELL_DISTANCE_WEIGHT,
    "CELL_TRAFFIC_WEIGHT": CELL_TRAFFIC_WEIGHT,
    "CELL_NEIGHBOR_BONUS": CELL_NEIGHBOR_BONUS,
    "CELL_UNCUT_WEIGHT": CELL_UNCUT_WEIGHT,
    "CELL_RETURN_HOME_WEIGHT": CELL_RETURN_HOME_WEIGHT,

    # Recovery
    "RECOVERY_MAX_TARGET_DIST": RECOVERY_MAX_TARGET_DIST,
    "RECOVERY_VISIT_PENALTY": RECOVERY_VISIT_PENALTY,
    "RECOVERY_CELL_SWITCH_PENALTY": RECOVERY_CELL_SWITCH_PENALTY,
    "RECOVERY_CANDIDATES": RECOVERY_CANDIDATES,

    # Energy / return
    "ENERGY_RESERVE": ENERGY_RESERVE,
    "RETURN_HOME_RATIO": RETURN_HOME_RATIO,

    # Watchdogs
    "NO_CUT_LIMIT": NO_CUT_LIMIT,
    "STUCK_LIMIT": STUCK_LIMIT,
    "LOOP_WINDOW": LOOP_WINDOW,
    "LOOP_UNIQUE_LIMIT": LOOP_UNIQUE_LIMIT,
    "LOOP_TRIGGER": LOOP_TRIGGER,

    # Benchmark early stop
    "MIN_TOTAL_REWARD": MIN_TOTAL_REWARD,
    "MIN_RECENT_REWARD": MIN_RECENT_REWARD,
    "RECENT_REWARD_WINDOW": RECENT_REWARD_WINDOW,
    "MAX_RECHARGES": MAX_RECHARGES,
    "SPIN_TURN_RATIO": SPIN_TURN_RATIO,
    "SPIN_MIN_COVERAGE": SPIN_MIN_COVERAGE,

    "USE_HIGH_LEVEL_POLICY": USE_HIGH_LEVEL_POLICY,
}
# =========================================
# LEGACY COMPATIBILITY FOR GLOBAL PLANNER
# =========================================

UNKNOWN_COST_AVOID = 1.0
UNKNOWN_COST_ALLOW = 0.3
UNKNOWN_COST_EXPLORE = 0.05

DYNAMIC_NEAR_COST = 2.0
DYNAMIC_MID_COST = 0.5
DYNAMIC_FAR_COST = 0.1

PREDICTION_HORIZON = 5
PREDICTION_COST = 0.4
PREDICTION_DECAY = 0.6

DYNAMIC_TRAFFIC_COST = 0.05

TURN_COST_WEIGHT = 0.35
DIRECTION_BIAS_WEIGHT = 2.0
BACKTRACK_PENALTY = 5.0
REPLAN_INTERVAL = 4

OPPORTUNISTIC_RETURN_MARGIN = 5.0
OPPORTUNISTIC_MAX_EXTRA_COST = 40.0
OPPORTUNISTIC_MIN_CABBAGES = 1
# =========================================
# ENERGY
# =========================================

MOVE_COST = 0.05
TURN_COST = 0.02
CUT_COST = 0.4

ENERGY_RESERVE = 30.0

# =========================================
# LAWNMOWER
# =========================================

ROBOT_W = 0.50
ROBOT_H = 0.50

CELL_SIZE_M = 0.25

CUT_RADIUS_CELLS = 1

GRASS_CELL = 1
CUT_CELL = 2
OBSTACLE_CELL = -1
BUFFER_CELL = 3

EDGE_SAFE_MODE = True
OBSTACLE_INFLATION_M = 0.35

# =========================================
# SECTORS
# =========================================

SECTOR_H = 10
SECTOR_W = 10

SECTOR_SCORE_CABBAGE_WEIGHT = 10.0
SECTOR_SCORE_TRAVEL_WEIGHT = 1.0
SECTOR_SCORE_ENERGY_WEIGHT = 1.0
SECTOR_SWITCH_PENALTY = 25.0


# =========================================
# COVERAGE
# =========================================

COVERAGE_SWEEP_MODE = "boustrophedon"

# =========================================
# REWARD
# =========================================

STEP_PENALTY = -0.05

OBSTACLE_PENALTY = -1.0
START_BLOCK_PENALTY = -0.5

COLLECT_REWARD = 10.0

RETURN_REWARD = 100.0
FINAL_BONUS = 50.0

ENERGY_REWARD_WEIGHT = 1.0

# =========================================
# MISSION
# =========================================

RETURN_LIMIT_MULT = 2

# =========================================
# RL
# =========================================

GAMMA = 0.99
TAU = 0.005

# =========================================
# TRAIN
# =========================================

EPISODES = 1000

BATCH_SIZE = 64
MEMORY_SIZE = 50000

# =========================================
# STATE
# =========================================

STATE_CHANNELS = 16

# =========================================
# DEVICE
# =========================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)