"""
Shared Electrical Helper Functions
===================================

Common electrical calculations used by both intra-array and inter-array
cost modules. All parameters are required (no hidden defaults).
"""

import numpy as np

from .coefficients import CABLE_COST_COEFFICIENTS, EUR_TO_USD


def calculate_mva(power_mw, power_factor):
    """
    Convert real power (MW) to apparent power (MVA).

    Args:
        power_mw: Real power in MW
        power_factor: Power factor (e.g. 0.95)

    Returns:
        Apparent power in MVA
    """
    return power_mw / power_factor


def calculate_current(mva, voltage_v):
    """
    Calculate current for given MVA and voltage (3-phase AC).

    Args:
        mva: Apparent power in MVA
        voltage_v: Line-to-line voltage in volts

    Returns:
        Current in amperes
    """
    return (mva * 1e6) / (np.sqrt(3) * voltage_v)


def calculate_cable_cost_per_km(mva, voltage_v):
    """
    Calculate cable cost using Collin 2017 exponential model.

    Model: cost (kEUR/km) = c1 + c2 * exp(c3 * S), where S = MVA rating

    Uses the unified CABLE_COST_COEFFICIENTS dict covering all voltages
    from 6.6 kV (intra-array) through 132 kV (inter-array).

    Args:
        mva: Cable MVA rating
        voltage_v: Cable voltage in volts

    Returns:
        Cable cost in $/km
    """
    if voltage_v not in CABLE_COST_COEFFICIENTS:
        raise ValueError(f"No coefficients for {voltage_v}V. "
                        f"Available: {sorted(CABLE_COST_COEFFICIENTS.keys())}")

    coeffs = CABLE_COST_COEFFICIENTS[voltage_v]
    c1, c2, c3 = coeffs['c1'], coeffs['c2'], coeffs['c3']

    # Calculate cost in kEUR/km, then convert to $/km
    cost_keur_km = c1 + c2 * np.exp(c3 * mva)
    return cost_keur_km * 1000 * EUR_TO_USD
