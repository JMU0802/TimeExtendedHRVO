# Core modules for Time-Extended HRVO

from .hrvo import HRVO, compute_hrvo, compute_vo, compute_rvo
from .strategy import AvoidanceStrategy
from .feasibility import is_strategy_feasible, compute_feasibility_margin
from .velocity_space import (
    is_in_hrvo_union, is_velocity_feasible_time_extended,
    sample_velocity_space, find_feasible_velocities,
    velocity_to_strategy, compute_velocity_cost,
    search_optimal_velocity, adaptive_velocity_search,
    visualize_velocity_space
)