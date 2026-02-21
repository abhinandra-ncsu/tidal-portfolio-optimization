"""
Cost Module
===========

Functions for calculating tidal project costs.

Main functions:
    - calculate_total_cost: Complete cost breakdown for a project
    - calculate_fixed_cost_per_array: Fixed costs per array
    - calculate_device_cost: Turbine lifecycle costs
    - calculate_transmission_cost: HVAC/HVDC transmission costs
    - calculate_inter_array_cost: Array to collection point cables
    - calculate_intra_array_cost: Within-array cables

Example:
    from tidal_portfolio.costs import calculate_total_cost

    costs = calculate_total_cost(
        num_arrays=3,
        inter_array_distances_km=[8.0, 10.0, 12.0],
        shore_distance_km=70.0,
        turbines_per_array=40,
        rated_power_mw=1.1,
        fcr=0.113,
        rows=5, cols=8,
        power_factor=0.95,
        row_spacing_km=0.300, col_spacing_km=0.080,
        intra_array_voltage_v=11000, inter_array_voltage_v=66000,
        array_power_mw=44.0,
        capacity_factor=0.35, opex_rate=0.025,
    )
    print(f"Total annual cost: ${costs['total_cost']:,.0f}/year")
"""

# Main aggregator functions
from .aggregator import (
    calculate_total_cost,
    calculate_fixed_cost_per_array,
)

# Device cost functions
from .device import (
    calculate_device_cost,
    calculate_capex_breakdown,
    calculate_opex_breakdown,
)

# Electrical infrastructure functions
from .electrical import (
    calculate_intra_array_cost,
    calculate_intra_array_cost_for_arrays,
    calculate_inter_array_cost,
    calculate_single_inter_array_cost,
    calculate_transmission_cost,
    calculate_hvac_cost,
    calculate_hvdc_cost,
)

__all__ = [
    # Main functions
    "calculate_total_cost",
    "calculate_fixed_cost_per_array",
    # Device
    "calculate_device_cost",
    "calculate_capex_breakdown",
    "calculate_opex_breakdown",
    # Electrical
    "calculate_intra_array_cost",
    "calculate_intra_array_cost_for_arrays",
    "calculate_inter_array_cost",
    "calculate_single_inter_array_cost",
    "calculate_transmission_cost",
    "calculate_hvac_cost",
    "calculate_hvdc_cost",
]
