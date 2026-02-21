"""
Inter-Array Cable and Transformer Cost Functions
=================================================

Cabling costs from turbine arrays to collection point using
Collin et al. (2017) empirical cost models.

Methodology: see writing/electrical-infrastructure/inter-array-cables.md
"""

import numpy as np

from .coefficients import (
    INTER_ARRAY_TRANSFORMER_COEFFICIENTS,
    SUBSTATION_ANCILLARY_COST,
)
from .helpers import calculate_mva, calculate_current, calculate_cable_cost_per_km


# =============================================================================
# COST CALCULATION FUNCTIONS
# =============================================================================

def calculate_substation_transformer_cost(mva, voltage_v):
    """
    Calculate array substation transformer cost (11 kV -> inter-array voltage).

    This transformer steps up from intra-array voltage (11 kV) to
    inter-array voltage (33/66/132 kV) at each array's collection point.

    Args:
        mva: Transformer MVA rating (typically full array power)
        voltage_v: Inter-array voltage in volts

    Returns:
        Transformer cost in $
    """
    # Select transformer coefficients based on voltage
    if voltage_v <= 33000:
        coeffs = INTER_ARRAY_TRANSFORMER_COEFFICIENTS['MV_33kV']
    elif voltage_v <= 66000:
        coeffs = INTER_ARRAY_TRANSFORMER_COEFFICIENTS['MV_66kV']
    else:
        coeffs = INTER_ARRAY_TRANSFORMER_COEFFICIENTS['MV_132kV']

    c1, c2, c3 = coeffs['c1'], coeffs['c2'], coeffs['c3']

    # Calculate cost (linear model: cost = c1 * S + c3)
    return c1 * (mva ** c2) + c3


# =============================================================================
# MAIN COST FUNCTIONS
# =============================================================================

def calculate_inter_array_cost(distances_km, array_power_mw, voltage_v,
                                power_factor, fcr, include_transformer=True,
                                substation_ancillary_cost=SUBSTATION_ANCILLARY_COST):
    """
    Calculate inter-array cable and transformer costs.

    Each array requires:
    1. A step-up transformer (11 kV -> 66 kV) at the array collection point
    2. A cable from the array to the central collection point
    3. Substation ancillary equipment (enclosure, switchgear, protection, installation)

    Args:
        distances_km: Distance from each array to collection point (list or array)
        array_power_mw: Power per array in MW
        voltage_v: Inter-array cable voltage in volts
        power_factor: Power factor
        fcr: Fixed Charge Rate
        include_transformer: Include array substation transformer cost (default: True)
        substation_ancillary_cost: Ancillary cost per substation in $
            (default: SUBSTATION_ANCILLARY_COST from coefficients)

    Returns:
        dict with cable costs, transformer costs, substation ancillary costs, and totals
    """
    distances = np.atleast_1d(distances_km)
    num_arrays = len(distances)
    array_mva = calculate_mva(array_power_mw, power_factor)
    current_a = calculate_current(array_mva, voltage_v)

    # -------------------------------------------------------------------------
    # Cable costs
    # -------------------------------------------------------------------------
    cable_cost_per_km = calculate_cable_cost_per_km(array_mva, voltage_v)

    cable_costs_by_array = distances * cable_cost_per_km
    total_cable_length = float(np.sum(distances))
    total_cable_cost = float(np.sum(cable_costs_by_array))

    # -------------------------------------------------------------------------
    # Transformer costs (one per array)
    # -------------------------------------------------------------------------
    if include_transformer:
        transformer_cost_each = calculate_substation_transformer_cost(array_mva, voltage_v)
        total_transformer_cost = transformer_cost_each * num_arrays
    else:
        transformer_cost_each = 0.0
        total_transformer_cost = 0.0

    # -------------------------------------------------------------------------
    # Substation ancillary costs (one per array)
    # Covers: subsea enclosure/platform, switchgear, protection, installation
    # -------------------------------------------------------------------------
    total_substation_ancillary = substation_ancillary_cost * num_arrays

    # -------------------------------------------------------------------------
    # Totals
    # -------------------------------------------------------------------------
    total_capex = total_cable_cost + total_transformer_cost + total_substation_ancillary
    annualized_cost = total_capex * fcr

    # Intra-array voltage (for documentation only)
    intra_array_voltage_v = 11000

    return {
        # Cable breakdown
        'cable': {
            'distances_km': list(distances),
            'total_length_km': total_cable_length,
            'cost_per_km': cable_cost_per_km,
            'costs_by_array': list(cable_costs_by_array),
            'total_capex': total_cable_cost,
            'voltage_v': voltage_v,
            'current_a': current_a,
        },
        # Transformer breakdown
        'transformer': {
            'num_transformers': num_arrays if include_transformer else 0,
            'mva_each': array_mva,
            'cost_each': transformer_cost_each,
            'total_capex': total_transformer_cost,
            'type': f'{intra_array_voltage_v/1000:.0f}kV_to_{voltage_v/1000:.0f}kV',
            'included': include_transformer,
        },
        # Substation ancillary breakdown
        'substation_ancillary': {
            'cost_each': substation_ancillary_cost,
            'num_substations': num_arrays,
            'total_capex': total_substation_ancillary,
        },
        # Totals
        'total_capex': total_capex,
        'annualized_cost': annualized_cost,
        'fcr': fcr,
        # Array info
        'num_arrays': num_arrays,
        'array_power_mw': array_power_mw,
        'array_mva': array_mva,
        'power_factor': power_factor,
        # Methodology documentation
        'methodology': {
            'cable_model': 'Collin et al. (2017) Table A1',
            'transformer_model': 'Collin et al. (2017) Table A3 (estimated)',
            'substation_ancillary_model': (
                'Lump-sum per substation covering subsea enclosure/platform, '
                'switchgear, protection systems, and installation. '
                'Smart & Noonan (2016); Nakhai et al. (2023)'
            ),
            'voltage_selection': (
                f'{voltage_v/1000:.0f} kV selected based on: '
                f'(1) array power = {array_power_mw:.1f} MW = {array_mva:.1f} MVA, '
                f'(2) break-even analysis shows 66 kV cheaper for distances > 2.1 km, '
                f'(3) typical inter-array distances are 15-25 km'
            ),
            'transformer_purpose': (
                f'Steps up from {intra_array_voltage_v/1000:.0f} kV (intra-array) '
                f'to {voltage_v/1000:.0f} kV (inter-array) at each array collection point'
            ),
        },
    }


def calculate_single_inter_array_cost(distance_km, array_power_mw, voltage_v,
                                       power_factor, fcr, include_transformer=True,
                                       substation_ancillary_cost=SUBSTATION_ANCILLARY_COST):
    """
    Calculate inter-array cost for a single array.

    Args:
        distance_km: Distance from array to collection point
        array_power_mw: Power per array in MW
        voltage_v: Inter-array voltage in volts
        power_factor: Power factor
        fcr: Fixed Charge Rate
        include_transformer: Include transformer cost (default: True)
        substation_ancillary_cost: Ancillary cost per substation in $
            (default: SUBSTATION_ANCILLARY_COST from coefficients)

    Returns:
        dict with cost breakdown
    """
    return calculate_inter_array_cost(
        [distance_km],
        array_power_mw=array_power_mw,
        voltage_v=voltage_v,
        power_factor=power_factor,
        fcr=fcr,
        include_transformer=include_transformer,
        substation_ancillary_cost=substation_ancillary_cost,
    )
