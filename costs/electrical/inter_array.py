"""
Inter-Array Cable and Transformer Cost Functions
=================================================

Cabling costs from turbine arrays to collection point using
Collin et al. (2017) empirical cost models.

Methodology: see writing/electrical-infrastructure/inter-array-cables.md
"""

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

def calculate_inter_array_cost(distance_km, array_power_mw, config,
                                include_transformer=True,
                                substation_ancillary_cost=SUBSTATION_ANCILLARY_COST):
    """
    Calculate inter-array cost for a single array-to-collection-point link.

    Cost of transferring energy from one array collection point to the
    offshore collection point, including:
    1. A step-up transformer (11 kV -> 66 kV) at the array collection point
    2. A cable from the array to the central collection point
    3. Substation ancillary equipment (enclosure, switchgear, protection, installation)

    Args:
        distance_km: Distance from array to collection point (km)
        array_power_mw: Power per array in MW
        config: ProjectConfig with inter_array_voltage_v, power_factor, fcr
        include_transformer: Include array substation transformer cost (default: True)
        substation_ancillary_cost: Ancillary cost per substation in $
            (default: SUBSTATION_ANCILLARY_COST from coefficients)

    Returns:
        dict with cable, transformer, substation ancillary, and total costs
    """
    voltage_v = config.inter_array_voltage_v
    power_factor = config.power_factor
    fcr = config.fcr

    array_mva = calculate_mva(array_power_mw, power_factor)
    current_a = calculate_current(array_mva, voltage_v)

    # Cable cost
    cable_cost_per_km = calculate_cable_cost_per_km(array_mva, voltage_v)
    cable_capex = distance_km * cable_cost_per_km

    # Transformer cost (one per array)
    if include_transformer:
        transformer_capex = calculate_substation_transformer_cost(array_mva, voltage_v)
    else:
        transformer_capex = 0.0

    # Totals
    total_capex = cable_capex + transformer_capex + substation_ancillary_cost
    annualized_cost = total_capex * fcr

    return {
        'cable': {
            'distance_km': distance_km,
            'cost_per_km': cable_cost_per_km,
            'capex': cable_capex,
            'voltage_v': voltage_v,
            'current_a': current_a,
        },
        'transformer': {
            'capex': transformer_capex,
            'mva': array_mva,
            'included': include_transformer,
        },
        'substation_ancillary': {
            'capex': substation_ancillary_cost,
        },
        'total_capex': total_capex,
        'annualized_cost': annualized_cost,
        'fcr': fcr,
        'array_power_mw': array_power_mw,
        'array_mva': array_mva,
        'power_factor': power_factor,
    }
