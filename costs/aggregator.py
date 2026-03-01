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

from .device import calculate_device_cost
from .electrical.intra_array import calculate_intra_array_cost_for_arrays
from .electrical.inter_array import calculate_inter_array_cost
from .electrical.transmission import calculate_transmission_cost


def calculate_total_cost(num_arrays, inter_array_distances_km, shore_distance_km,
                          rated_power_mw, array_power_mw, capacity_factor, config):
    """
    Calculate complete cost breakdown for a tidal project.

    Args:
        num_arrays: Number of arrays
        inter_array_distances_km: Distance from each array to collection point
        shore_distance_km: Distance from collection point to shore
        rated_power_mw: Turbine rated power in MW
        array_power_mw: Total power per array in MW
        capacity_factor: Average capacity factor
        config: ProjectConfig with all engineering constants

    Returns:
        dict with all cost components and totals
    """
    turbines_per_array = config.turbines_per_array
    fcr = config.fcr

    # Calculate project capacity
    project_capacity_mw = num_arrays * turbines_per_array * rated_power_mw

    # Device costs (for all arrays)
    total_turbines = num_arrays * turbines_per_array
    device = calculate_device_cost(total_turbines, fcr=fcr)

    # Intra-array costs (for all arrays)
    intra = calculate_intra_array_cost_for_arrays(
        num_arrays, config, turbine_power_mw=rated_power_mw,
    )

    # Inter-array costs (one per array)
    inter_costs = [
        calculate_inter_array_cost(d, array_power_mw=array_power_mw, config=config)
        for d in inter_array_distances_km
    ]
    inter_annualized = sum(c['annualized_cost'] for c in inter_costs)
    inter_capex = sum(c['total_capex'] for c in inter_costs)

    # Transmission costs
    trans = calculate_transmission_cost(
        shore_distance_km, project_capacity_mw,
        capacity_factor=capacity_factor, config=config,
    )

    # Totals
    fixed_cost = device['annualized_cost'] + intra['annualized_cost']
    variable_cost = inter_annualized + trans['annualized_cost']
    total_cost = fixed_cost + variable_cost

    total_capex = (device['total_capex'] + intra['total_capex'] +
                   inter_capex + trans['capex'])

    return {
        # Summary
        'total_cost': total_cost,
        'fixed_cost': fixed_cost,
        'variable_cost': variable_cost,
        'total_capex': total_capex,

        # Component costs (annualized)
        'device_cost': device['annualized_cost'],
        'intra_array_cost': intra['annualized_cost'],
        'inter_array_cost': inter_annualized,
        'transmission_cost': trans['annualized_cost'],

        # Transmission info
        'transmission_mode': trans['mode'],
        'transmission_efficiency': trans['efficiency'],

        # Component details
        'device': device,
        'intra_array': intra,
        'inter_array': inter_costs,
        'transmission': trans,

        # Project info
        'num_arrays': num_arrays,
        'turbines_per_array': turbines_per_array,
        'total_turbines': total_turbines,
        'project_capacity_mw': project_capacity_mw,
        'fcr': fcr,
    }


def calculate_total_fixed_cost(num_arrays, turbine_power_mw, config):
    """
    Calculate total fixed cost for the entire fleet (device + intra-array).

    Computes device cost for all turbines at once to capture economies of
    scale (sublinear power laws, fixed infrastructure overheads).

    Args:
        num_arrays: Number of arrays in the fleet
        turbine_power_mw: Power per turbine in MW
        config: ProjectConfig with turbines_per_array, fcr, and array layout

    Returns:
        Total fixed cost for the fleet in $/year
    """
    total_turbines = num_arrays * config.turbines_per_array
    device = calculate_device_cost(total_turbines, fcr=config.fcr)
    intra = calculate_intra_array_cost_for_arrays(
        num_arrays, config, turbine_power_mw=turbine_power_mw,
    )

    return device['annualized_cost'] + intra['annualized_cost']
