"""
Tidal Portfolio Optimization Package
=====================================

A modular framework for optimizing tidal turbine array portfolios.

Modules:
    - site_processing: HYCOM data → feasible sites
    - costs: Device, intra-array, inter-array, and transmission costs
    - energy: Generation, wake losses, and covariance calculations
    - optimization: Portfolio optimization model and solver

Example:
    from tidal_portfolio import load_turbine, run_portfolio_optimization, get_best_result
    from tidal_portfolio.config import TURBINES_PER_ARRAY, WAKE_LOSS_FACTOR, FCR

    # Load turbine spec
    turbine = load_turbine("RM1")

    # Run optimization (site_data from load_all → flatten → process_sites)
    results = run_portfolio_optimization(
        site_data=site_data,
        num_arrays=3,
        lcoe_targets=[700, 750, 800],
        wake_loss_factor=WAKE_LOSS_FACTOR,
        turbines_per_array=TURBINES_PER_ARRAY,
        rated_power_mw=turbine["rated_power_mw"],
    )

    # Get best result
    best = get_best_result(results['results'])
    print(f"LCOE: ${best['achieved_lcoe']:.0f}/MWh")
    print(f"Variance: {best['variance']:.2f} MW²")
"""

__version__ = "4.0.0"
__author__ = "Tidal Energy Project"

# Turbine loading
from .site_processing.turbine import load_turbine, list_available_turbines

# Main optimization functions
from .optimization import (
    run_portfolio_optimization,
    calculate_distance_matrix,
    get_best_result,
    get_feasible_count,
    save_optimization_results,
)

# Cost functions
from .costs import (
    calculate_total_cost,
    calculate_fixed_cost_per_array,
    calculate_device_cost,
    calculate_transmission_cost,
    calculate_inter_array_cost,
    calculate_intra_array_cost,
)

# Energy functions
from .energy import (
    prepare_energy_data,
    calculate_energy_vector,
    calculate_covariance,
    apply_power_curve,
)

# Site processing functions
from .site_processing import (
    process_sites,
    load_all,
    flatten_grid_data,
    load_site_results,
)

# Visualization functions
from .visualization import (
    plot_all,
    plot_lcoe_vs_variance,
    plot_site_map,
    plot_cost_breakdown,
    plot_capacity_factor_comparison,
    plot_site_selection_frequency,
    plot_energy_vs_variance,
)

__all__ = [
    # Turbine loading
    "load_turbine",
    "list_available_turbines",
    # Main optimization functions
    "run_portfolio_optimization",
    "calculate_distance_matrix",
    "get_best_result",
    "get_feasible_count",
    "save_optimization_results",
    # Cost functions
    "calculate_total_cost",
    "calculate_fixed_cost_per_array",
    "calculate_device_cost",
    "calculate_transmission_cost",
    "calculate_inter_array_cost",
    "calculate_intra_array_cost",
    # Energy functions
    "prepare_energy_data",
    "calculate_energy_vector",
    "calculate_covariance",
    "apply_power_curve",
    # Site processing functions
    "process_sites",
    "load_all",
    "flatten_grid_data",
    "load_site_results",
    # Visualization functions
    "plot_all",
    "plot_lcoe_vs_variance",
    "plot_site_map",
    "plot_cost_breakdown",
    "plot_capacity_factor_comparison",
    "plot_site_selection_frequency",
    "plot_energy_vs_variance",
]
