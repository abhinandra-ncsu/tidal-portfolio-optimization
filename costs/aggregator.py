"""
Total Cost Functions
====================

Functions for combining all cost components into total project cost.

Components:
    1. Device costs (turbines)
    2. Intra-array cables (within array)
    3. Inter-array cables (array to collection point)
    4. Transmission (collection point to shore)
"""

import numpy as np

from .device import calculate_device_cost
from .electrical.intra_array import calculate_intra_array_cost_for_arrays
from .electrical.inter_array import calculate_inter_array_cost
from .electrical.transmission import calculate_transmission_cost


def calculate_total_cost(num_arrays, inter_array_distances_km, shore_distance_km,
                          turbines_per_array, rated_power_mw, fcr,
                          rows, cols, power_factor,
                          row_spacing_km, col_spacing_km,
                          intra_array_voltage_v, inter_array_voltage_v,
                          array_power_mw, capacity_factor, opex_rate):
    """
    Calculate complete cost breakdown for a tidal project.

    Args:
        num_arrays: Number of arrays
        inter_array_distances_km: Distance from each array to collection point
        shore_distance_km: Distance from collection point to shore
        turbines_per_array: Turbines per array
        rated_power_mw: Turbine rated power in MW
        fcr: Fixed Charge Rate (e.g., 0.113 = 11.3%)
        rows: Array rows
        cols: Array columns
        power_factor: Power factor
        row_spacing_km: Row spacing in km
        col_spacing_km: Column spacing in km
        intra_array_voltage_v: Intra-array cable voltage in volts
        inter_array_voltage_v: Inter-array cable voltage in volts
        array_power_mw: Total power per array in MW
        capacity_factor: Average capacity factor
        opex_rate: OPEX as fraction of CAPEX

    Returns:
        dict with all cost components and totals
    """
    # Calculate project capacity
    project_capacity_mw = num_arrays * turbines_per_array * rated_power_mw

    # Device costs (for all arrays)
    total_turbines = num_arrays * turbines_per_array
    device = calculate_device_cost(total_turbines, fcr=fcr)

    # Intra-array costs (for all arrays)
    intra = calculate_intra_array_cost_for_arrays(
        num_arrays, rows=rows, cols=cols,
        turbine_power_mw=rated_power_mw, power_factor=power_factor,
        row_spacing_km=row_spacing_km, col_spacing_km=col_spacing_km,
        voltage_v=intra_array_voltage_v, fcr=fcr,
    )

    # Inter-array costs
    inter = calculate_inter_array_cost(
        inter_array_distances_km,
        array_power_mw=array_power_mw, voltage_v=inter_array_voltage_v,
        power_factor=power_factor, fcr=fcr,
    )

    # Transmission costs
    trans = calculate_transmission_cost(
        shore_distance_km, project_capacity_mw,
        capacity_factor=capacity_factor, fcr=fcr, opex_rate=opex_rate,
    )

    # Totals
    fixed_cost = device['annualized_cost'] + intra['annualized_cost']
    variable_cost = inter['annualized_cost'] + trans['annualized_cost']
    total_cost = fixed_cost + variable_cost

    total_capex = (device['total_capex'] + intra['total_capex'] +
                   inter['total_capex'] + trans['capex'])

    return {
        # Summary
        'total_cost': total_cost,
        'fixed_cost': fixed_cost,
        'variable_cost': variable_cost,
        'total_capex': total_capex,

        # Component costs (annualized)
        'device_cost': device['annualized_cost'],
        'intra_array_cost': intra['annualized_cost'],
        'inter_array_cost': inter['annualized_cost'],
        'transmission_cost': trans['annualized_cost'],

        # Transmission info
        'transmission_mode': trans['mode'],
        'transmission_efficiency': trans['efficiency'],

        # Component details
        'device': device,
        'intra_array': intra,
        'inter_array': inter,
        'transmission': trans,

        # Project info
        'num_arrays': num_arrays,
        'turbines_per_array': turbines_per_array,
        'total_turbines': total_turbines,
        'project_capacity_mw': project_capacity_mw,
        'fcr': fcr,
    }


def calculate_total_fixed_cost(num_arrays, turbines_per_array, fcr, rows, cols,
                                turbine_power_mw, power_factor,
                                row_spacing_km, col_spacing_km,
                                intra_array_voltage_v):
    """
    Calculate total fixed cost for the entire fleet (device + intra-array).

    Computes device cost for all turbines at once to capture economies of
    scale (sublinear power laws, fixed infrastructure overheads).

    Args:
        num_arrays: Number of arrays in the fleet
        turbines_per_array: Turbines per array
        fcr: Fixed Charge Rate (e.g., 0.113 = 11.3%)
        rows: Array rows
        cols: Array columns
        turbine_power_mw: Power per turbine in MW
        power_factor: Power factor
        row_spacing_km: Row spacing in km
        col_spacing_km: Column spacing in km
        intra_array_voltage_v: Intra-array cable voltage in volts

    Returns:
        Total fixed cost for the fleet in $/year
    """
    total_turbines = num_arrays * turbines_per_array
    device = calculate_device_cost(total_turbines, fcr=fcr)
    intra = calculate_intra_array_cost_for_arrays(
        num_arrays, rows=rows, cols=cols,
        turbine_power_mw=turbine_power_mw, power_factor=power_factor,
        row_spacing_km=row_spacing_km, col_spacing_km=col_spacing_km,
        voltage_v=intra_array_voltage_v, fcr=fcr,
    )

    return device['annualized_cost'] + intra['annualized_cost']
