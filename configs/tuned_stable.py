"""
Stable tuned lawnmower profile.
Auto-tune winner.
"""
PROFILE_NAME = "tuned_stable_v2_adaptive"

# =========================================
# COVERAGE TRAFFIC
# =========================================

VISIT_WEIGHT = 0.03
CUT_WEIGHT = 0.06
TRAFFIC_MAX_PENALTY = 8.0

# =========================================
# CELL ORDERING
# =========================================

CELL_TRAFFIC_WEIGHT = 0.6
CELL_NEIGHBOR_BONUS = 20.0
CELL_DISTANCE_WEIGHT = 1.5

# =========================================
# ENERGY
# =========================================

ENERGY_RESERVE = 28.0

