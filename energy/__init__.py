"""
Energy Module
=============

Functions for energy generation and portfolio variance calculations.

Main functions:
    - apply_power_curve: Convert current speeds to power output
    - calculate_capacity_factor: Get capacity factor from power timeseries
    - calculate_energy_vector: Get annual energy for multiple sites
    - calculate_covariance: Get covariance matrix for portfolio optimization
    - prepare_energy_data: Convenience function to prepare all energy data

Example:
    from tidal_portfolio.energy import prepare_energy_data, calculate_covariance

    # Prepare all energy data for optimization
    energy_data = prepare_energy_data(
        capacity_factors=site_cf,
        power_timeseries=site_power,
        rated_power_mw=1.1,
        turbines_per_array=40,
        wake_loss_factor=0.88,
    )

    # Access results
    energy_vector = energy_data['energy_vector']
    cov_matrix = energy_data['covariance_matrix']
"""

# Generation functions
from .generation import (
    apply_power_curve,
    calculate_capacity_factor,
    calculate_energy_vector,
)

# Covariance functions
from .covariance import (
    calculate_covariance,
    get_covariance_subset,
    calculate_portfolio_variance,
)

# Convenience functions
from .model import (
    prepare_energy_data,
    get_covariance_for_sites,
)

__all__ = [
    # Generation
    "apply_power_curve",
    "calculate_capacity_factor",
    "calculate_energy_vector",
    # Covariance
    "calculate_covariance",
    "get_covariance_subset",
    "calculate_portfolio_variance",
    # Convenience
    "prepare_energy_data",
    "get_covariance_for_sites",
]
