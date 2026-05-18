import torch

# =========================================
# CONFIG VERSION
# =========================================

CONFIG_VERSION = "wait_autotune_v1"

# ========================================
# FLAGS
# ========================================
USE_LOCAL_RL = False
USE_HIERARCHICAL_PLANNER = False
USE_PORTAL_PLANNER = False

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
# AUTO-TUNED PARAMETERS
# =========================================

TURN_COST_WEIGHT = 0.35

UNKNOWN_COST_AVOID = 1.0
UNKNOWN_COST_ALLOW = 0.3
UNKNOWN_COST_EXPLORE = 0.05

DYNAMIC_NEAR_COST = 2.0
DYNAMIC_MID_COST = 0.5
DYNAMIC_FAR_COST = 0.1

REPLAN_INTERVAL = 4

DIRECTION_BIAS_WEIGHT = 2.0
BACKTRACK_PENALTY = 5.0

STRAIGHT_BONUS = 0.01
TURN_CHANGE_PENALTY = 0.03

LOCAL_TARGET_RADIUS = 4
LOCAL_TARGET_BONUS = 2.0
SWEEP_STICKINESS = 1.0

OPPORTUNISTIC_RETURN_MARGIN = 5.0
OPPORTUNISTIC_MAX_EXTRA_COST = 40.0
OPPORTUNISTIC_MIN_CABBAGES = 1

PREDICTION_HORIZON = 5
PREDICTION_COST = 0.4
PREDICTION_DECAY = 0.6

DYNAMIC_TRAFFIC_COST = 0.05

# =========================================
# RUNTIME TUNABLES
# =========================================

TUNABLES = {

    "TURN_COST_WEIGHT": TURN_COST_WEIGHT,
    "UNKNOWN_COST_AVOID": UNKNOWN_COST_AVOID,
    "UNKNOWN_COST_ALLOW": UNKNOWN_COST_ALLOW,
    "UNKNOWN_COST_EXPLORE": UNKNOWN_COST_EXPLORE,
    "DYNAMIC_NEAR_COST": DYNAMIC_NEAR_COST,
    "DYNAMIC_MID_COST": DYNAMIC_MID_COST,
    "DYNAMIC_FAR_COST": DYNAMIC_FAR_COST,
    "REPLAN_INTERVAL": REPLAN_INTERVAL,
    "DIRECTION_BIAS_WEIGHT": DIRECTION_BIAS_WEIGHT,
    "BACKTRACK_PENALTY": BACKTRACK_PENALTY,
    "STRAIGHT_BONUS": STRAIGHT_BONUS,
    "TURN_CHANGE_PENALTY": TURN_CHANGE_PENALTY,
    "LOCAL_TARGET_RADIUS": LOCAL_TARGET_RADIUS,
    "LOCAL_TARGET_BONUS": LOCAL_TARGET_BONUS,
    "SWEEP_STICKINESS": SWEEP_STICKINESS,
    "OPPORTUNISTIC_RETURN_MARGIN": OPPORTUNISTIC_RETURN_MARGIN,
    "OPPORTUNISTIC_MAX_EXTRA_COST": OPPORTUNISTIC_MAX_EXTRA_COST,
    "OPPORTUNISTIC_MIN_CABBAGES": OPPORTUNISTIC_MIN_CABBAGES,
    "PREDICTION_HORIZON": PREDICTION_HORIZON,
    "PREDICTION_COST": PREDICTION_COST,
    "PREDICTION_DECAY": PREDICTION_DECAY,
    "DYNAMIC_TRAFFIC_COST": DYNAMIC_TRAFFIC_COST,
}

# =========================================
# ENERGY
# =========================================

MOVE_COST = 0.05
TURN_COST = 0.02
CUT_COST = 0.4

ENERGY_RESERVE = 5.0

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