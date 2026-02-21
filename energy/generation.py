"""
Energy Generation Functions
===========================

Functions for calculating tidal turbine energy generation.

Includes:
    - Power curve application (current speed → power)
    - Capacity factor calculation
    - Annual energy production for multiple sites (vectorized)
"""

import numpy as np

from ..config import HOURS_PER_YEAR


def apply_power_curve(current_speeds, cut_in_speed, rated_speed, cut_out_speed, rated_power_mw):
    """
    Apply turbine power curve to current speed timeseries.

    Power curve regions:
        - Below cut-in: 0
        - Cut-in to rated: cubic ramp P = P_rated × (v/v_rated)³
        - Rated to cut-out: rated power
        - Above cut-out: 0

    Args:
        current_speeds: Array of current speeds in m/s
        cut_in_speed: Cut-in speed in m/s
        rated_speed: Rated speed in m/s
        cut_out_speed: Cut-out speed in m/s
        rated_power_mw: Rated power in MW

    Returns:
        Array of power outputs in MW
    """
    v = np.asarray(current_speeds)
    power = np.zeros_like(v)

    # Region 2: Cut-in to rated (cubic curve)
    mask_ramp = (v >= cut_in_speed) & (v < rated_speed)
    power[mask_ramp] = rated_power_mw * (v[mask_ramp] / rated_speed) ** 3

    # Region 3: Rated to cut-out (constant)
    mask_rated = (v >= rated_speed) & (v <= cut_out_speed)
    power[mask_rated] = rated_power_mw

    return power


def calculate_capacity_factor(current_speeds, cut_in_speed, rated_speed, cut_out_speed, rated_power_mw):
    """
    Calculate capacity factor from current speed timeseries.

    Args:
        current_speeds: Array of current speeds in m/s
        cut_in_speed: Cut-in speed in m/s
        rated_speed: Rated speed in m/s
        cut_out_speed: Cut-out speed in m/s
        rated_power_mw: Rated power in MW

    Returns:
        Capacity factor (0-1 scale)
    """
    power = apply_power_curve(current_speeds, cut_in_speed, rated_speed, cut_out_speed, rated_power_mw)
    mean_power = np.mean(power)
    return mean_power / rated_power_mw


def calculate_energy_vector(capacity_factors, rated_power_mw, turbines_per_array, wake_loss_factor):
    """
    Calculate net annual energy for one or more sites (vectorized).

    Formula:
        E_gross = n_turbines × P_rated × CF × hours_per_year
        E_net = E_gross × wake_loss_factor

    Args:
        capacity_factors: Capacity factor(s) (0-1 scale), scalar or array
        rated_power_mw: Turbine rated power in MW
        turbines_per_array: Number of turbines per array
        wake_loss_factor: Wake loss multiplier (e.g., 0.88 = 12% loss)

    Returns:
        Net annual energy in MWh (scalar or array matching input)
    """
    capacity_factors = np.asarray(capacity_factors)
    gross = turbines_per_array * rated_power_mw * capacity_factors * HOURS_PER_YEAR
    return gross * wake_loss_factor
