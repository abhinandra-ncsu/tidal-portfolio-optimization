"""
Energy Module
=============

Functions for energy generation and portfolio variance calculations.

Main functions:
    - apply_power_curve: Convert current speeds to power output
    - calculate_capacity_factor: Get capacity factor from power timeseries
    - calculate_energy_vector: Get annual energy for multiple sites
    - calculate_covariance: Get covariance matrix for portfolio optimization
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

__all__ = [
    # Generation
    "apply_power_curve",
    "calculate_capacity_factor",
    "calculate_energy_vector",
    # Covariance
    "calculate_covariance",
    "get_covariance_subset",
    "calculate_portfolio_variance",
]
