"""
Optimization Module
===================

Functions for portfolio optimization of tidal turbine arrays.

Main Functions:
    - run_portfolio_optimization: Main entry point
    - solve_optimization_model: Solve single optimization

Example:
    from tidal_portfolio.optimization import run_portfolio_optimization

    results = run_portfolio_optimization(
        site_data=site_data,
        num_arrays=3,
        lcoe_targets=[700, 750, 800],
        wake_loss_factor=0.88,
    )
"""

# Main functions
from .portfolio import (
    run_portfolio_optimization,
    calculate_distance_matrix,
    find_viable_collection_points,
    optimize_for_lcoe_target,
    get_best_result,
    get_feasible_count,
)

# Solver functions
from .solver import (
    solve_optimization_model,
    solve_with_gurobi,
)

# Model building functions
from .model import (
    validate_model_inputs,
    build_optimization_model,
    get_model_solution,
    get_model_objective_value,
    evaluate_solution,
)

# Result saving
from .save_results import save_optimization_results

__all__ = [
    # Main functions
    "run_portfolio_optimization",
    "calculate_distance_matrix",
    "find_viable_collection_points",
    "optimize_for_lcoe_target",
    "get_best_result",
    "get_feasible_count",
    # Solver functions
    "solve_optimization_model",
    "solve_with_gurobi",
    # Model building functions
    "validate_model_inputs",
    "build_optimization_model",
    "get_model_solution",
    "get_model_objective_value",
    "evaluate_solution",
    # Result saving
    "save_optimization_results",
]
