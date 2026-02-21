"""
Covariance Functions
====================

Functions for calculating covariance matrix for portfolio variance optimization.

The covariance matrix captures how energy generation at different
sites co-varies over time. Sites with low covariance provide
diversification benefits when combined in a portfolio.
"""

import numpy as np


def calculate_covariance(power_timeseries, turbines_per_array, wake_loss_factor, scaled=True):
    """
    Calculate covariance matrix from power timeseries.

    The covariance captures temporal correlation between sites:
        - High covariance: Sites generate similarly over time
        - Low covariance: Sites provide diversification

    Scaling:
        Raw covariance is computed from single-turbine power.
        For array-level optimization, this is scaled by:
            Cov_array = Cov_turbine × (n_turbines × wake_factor)²

    Args:
        power_timeseries: Power output timeseries (n_sites × n_timesteps) in MW
        turbines_per_array: Turbines per array for scaling
        wake_loss_factor: Wake loss factor for scaling (e.g., 0.88 = 12% loss)
        scaled: If True, scale for array-level output (default: True)

    Returns:
        dict with:
            - covariance_matrix: (n_sites × n_sites) covariance matrix
            - n_sites: Number of sites
            - n_timesteps: Number of timesteps
            - scaled: Whether the matrix was scaled
    """
    power_timeseries = np.asarray(power_timeseries)

    # Validate input
    if power_timeseries.ndim != 2:
        raise ValueError(
            f"power_timeseries must be 2D (n_sites × n_timesteps), "
            f"got shape {power_timeseries.shape}"
        )

    n_sites, n_timesteps = power_timeseries.shape

    # Compute raw covariance matrix
    cov_matrix = np.cov(power_timeseries, rowvar=True)

    # Handle single-site case
    if cov_matrix.ndim == 0:
        cov_matrix = cov_matrix.reshape(1, 1)

    # Scale for array-level if requested
    if scaled:
        scale = turbines_per_array * wake_loss_factor
        cov_matrix = cov_matrix * (scale ** 2)

    return {
        'covariance_matrix': cov_matrix,
        'n_sites': n_sites,
        'n_timesteps': n_timesteps,
        'scaled': scaled,
    }


def get_covariance_subset(covariance_matrix, site_indices):
    """
    Extract covariance submatrix for selected sites.

    Args:
        covariance_matrix: Full covariance matrix (n × n)
        site_indices: Indices of selected sites

    Returns:
        Submatrix for selected sites
    """
    idx = np.ix_(site_indices, site_indices)
    return covariance_matrix[idx]


def calculate_portfolio_variance(covariance_matrix, weights):
    """
    Calculate portfolio variance given covariance matrix and weights.

    Formula: variance = w' × Σ × w

    Args:
        covariance_matrix: Covariance matrix (n × n)
        weights: Portfolio weights (n,) - typically binary for site selection

    Returns:
        Portfolio variance
    """
    weights = np.asarray(weights)
    return float(weights @ covariance_matrix @ weights)
