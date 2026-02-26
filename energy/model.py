"""
Energy Model Functions
======================

Convenience functions that combine generation, wake losses, and covariance
calculations for the optimization module.

These functions provide a simpler interface when you have all the data ready.
"""

import numpy as np

from .generation import calculate_energy_vector
from .covariance import calculate_covariance, get_covariance_subset


def prepare_energy_data(capacity_factors, power_timeseries, rated_power_mw,
                        turbines_per_array, wake_loss_factor):
    """
    Prepare all energy-related data needed for optimization.

    This is a convenience function that calculates energy vectors and
    covariance matrices in one call.

    Args:
        capacity_factors: Capacity factors for all sites (0-1 scale)
        power_timeseries: Power timeseries (n_sites × n_timesteps) in MW
        rated_power_mw: Turbine rated power in MW
        turbines_per_array: Number of turbines per array
        wake_loss_factor: Wake loss multiplier (e.g., 0.88 = 12% loss)

    Returns:
        dict with:
            - energy_vector: Net annual energy per site (MWh)
            - covariance_matrix: Scaled covariance matrix
            - n_sites: Number of sites
            - n_timesteps: Number of timesteps
            - array_capacity_mw: Array capacity in MW
    """
    capacity_factors = np.asarray(capacity_factors)
    power_timeseries = np.asarray(power_timeseries)

    # Calculate energy vector
    energy_vector = calculate_energy_vector(
        capacity_factors, rated_power_mw, turbines_per_array, wake_loss_factor
    )

    # Calculate covariance
    cov_result = calculate_covariance(
        power_timeseries, turbines_per_array, wake_loss_factor, scaled=True
    )

    return {
        'energy_vector': energy_vector,
        'covariance_matrix': cov_result['covariance_matrix'],
        'n_sites': len(capacity_factors),
        'n_timesteps': cov_result['n_timesteps'],
        'array_capacity_mw': turbines_per_array * rated_power_mw,
        'wake_loss_factor': wake_loss_factor,
    }


def get_covariance_for_sites(covariance_matrix, site_indices=None):
    """
    Get covariance matrix for selected sites.

    Args:
        covariance_matrix: Full covariance matrix
        site_indices: Indices of selected sites (None = all sites)

    Returns:
        Covariance matrix for selected sites
    """
    if site_indices is None:
        return covariance_matrix.copy()
    return get_covariance_subset(covariance_matrix, site_indices)
