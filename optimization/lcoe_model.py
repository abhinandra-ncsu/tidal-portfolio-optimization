"""
Enumeration-Based Optimization Models
=======================================

Min-LCOE and Max-Generation site selection via brute-force enumeration.
Shares CP-iteration logic through the enumeration_solver helpers.
"""

import numpy as np

from ..costs import calculate_total_fixed_cost
from ..energy import calculate_energy_vector, calculate_covariance
from .portfolio import calculate_distance_matrix, find_viable_collection_points
from .enumeration_solver import find_best_subset, prepare_cluster_inputs


def _prepare_shared_data(site_data, num_arrays, rated_power_mw,
                         cluster_radius_km, current_mode, config, verbose):
    """Compute all shared inputs needed by any optimization model."""
    array_power_mw = config.turbines_per_array * rated_power_mw
    project_capacity_mw = num_arrays * array_power_mw

    if verbose:
        print("Calculating distance matrix...")
    distance_matrix = calculate_distance_matrix(
        site_data['latitudes'], site_data['longitudes'],
    )

    if current_mode == "tidal":
        if 'tidal_capacity_factors' not in site_data or 'tidal_power_timeseries' not in site_data:
            raise ValueError(
                "current_mode='tidal' requires 'tidal_capacity_factors' and "
                "'tidal_power_timeseries' in site_data."
            )
        active_cf = site_data['tidal_capacity_factors']
        active_power = site_data['tidal_power_timeseries']
    else:
        active_cf = site_data['capacity_factors']
        active_power = site_data['power_timeseries']

    if verbose:
        mode_label = "tidal-only" if current_mode == "tidal" else "total"
        print(f"Preparing energy data (current_mode={mode_label})...")
    energy_vector = calculate_energy_vector(active_cf, rated_power_mw, config)
    cov_result = calculate_covariance(active_power, config, scaled=True)
    covariance_matrix = cov_result['covariance_matrix']

    if verbose:
        print("Finding viable collection points...")
    viable_cps = find_viable_collection_points(
        distance_matrix, cluster_radius_km, num_arrays,
    )
    if verbose:
        print(f"  Found {len(viable_cps)} viable collection points")

    if len(viable_cps) == 0:
        raise ValueError(
            f"No viable collection points found with radius {cluster_radius_km} km"
        )

    total_fixed_cost = calculate_total_fixed_cost(
        num_arrays=num_arrays,
        turbine_power_mw=rated_power_mw,
        config=config,
    )

    return {
        'distance_matrix': distance_matrix,
        'energy_vector': energy_vector,
        'covariance_matrix': covariance_matrix,
        'viable_cps': viable_cps,
        'total_fixed_cost': total_fixed_cost,
        'array_power_mw': array_power_mw,
        'project_capacity_mw': project_capacity_mw,
    }


def _optimize_by_enumeration(site_data, energy_vector, covariance_matrix,
                              distance_matrix, viable_cps, num_arrays,
                              cluster_radius_km, total_fixed_cost,
                              project_capacity_mw, array_power_mw,
                              config, objective, verbose=True):
    """
    Iterate over CPs, enumerate subsets, pick best by objective.

    Args:
        objective: "min_lcoe" or "max_generation"

    Returns:
        Result dict with keys: feasible, selected_sites, collection_point,
        variance, lcoe, total_cost, total_energy, cost_breakdown,
        transmission_mode, dist_to_shore_km, n_tried.
    """
    best_val = float('inf')
    best_entry = None
    best_cp = None
    best_global_indices = None
    best_trans_mode = None
    best_shore_dist = None

    n_tried = 0

    for cp_idx in viable_cps:
        cluster = prepare_cluster_inputs(
            cp_idx, distance_matrix, cluster_radius_km,
            energy_vector, covariance_matrix, site_data,
            project_capacity_mw, array_power_mw, config,
        )
        candidates, energy_subset, cov_subset, inter_array_costs, \
            transmission_cost, trans_mode, shore_dist = cluster

        if len(candidates) < num_arrays:
            continue
        n_tried += 1

        entry = find_best_subset(
            n_candidates=len(candidates),
            num_arrays=num_arrays,
            energy_per_site=energy_subset,
            covariance_matrix=cov_subset,
            total_fixed_cost=total_fixed_cost,
            inter_array_costs=inter_array_costs,
            transmission_cost=transmission_cost,
            objective=objective,
        )

        if entry is None:
            continue

        val = entry['lcoe'] if objective == "min_lcoe" else -entry['total_energy']
        if val < best_val:
            best_val = val
            best_entry = entry
            best_cp = cp_idx
            best_global_indices = candidates[entry['indices']]
            best_trans_mode = trans_mode
            best_shore_dist = shore_dist

    if best_entry is not None:
        label = (f"Min-LCOE: ${best_entry['lcoe']:.0f}/MWh"
                 if objective == "min_lcoe"
                 else f"Max-Gen: {best_entry['total_energy']:,.0f} MWh/year")
        if verbose:
            print(f"  {label}  (tried {n_tried} CPs)")

        return {
            'feasible': True,
            'selected_sites': best_global_indices,
            'collection_point': best_cp,
            'variance': best_entry['variance'],
            'lcoe': best_entry['lcoe'],
            'total_cost': best_entry['total_cost'],
            'total_energy': best_entry['total_energy'],
            'cost_breakdown': best_entry['cost_breakdown'],
            'transmission_mode': best_trans_mode,
            'dist_to_shore_km': best_shore_dist,
            'n_tried': n_tried,
        }
    else:
        if verbose:
            print(f"  No feasible subset found (tried {n_tried} CPs)")
        return {'feasible': False, 'n_tried': n_tried}


def _run_enumeration_model(site_data, num_arrays, rated_power_mw,
                            cluster_radius_km, current_mode, config,
                            objective, model_name, verbose):
    """Shared entry point for enumeration-based models."""
    from ..config import DEFAULT_CONFIG
    if config is None:
        config = DEFAULT_CONFIG

    data = _prepare_shared_data(
        site_data, num_arrays, rated_power_mw,
        cluster_radius_km, current_mode, config, verbose,
    )

    if verbose:
        label = "min-LCOE" if objective == "min_lcoe" else "max-generation"
        print(f"Running {label} enumeration...")

    result = _optimize_by_enumeration(
        site_data=site_data,
        energy_vector=data['energy_vector'],
        covariance_matrix=data['covariance_matrix'],
        distance_matrix=data['distance_matrix'],
        viable_cps=data['viable_cps'],
        num_arrays=num_arrays,
        cluster_radius_km=cluster_radius_km,
        total_fixed_cost=data['total_fixed_cost'],
        project_capacity_mw=data['project_capacity_mw'],
        array_power_mw=data['array_power_mw'],
        config=config,
        objective=objective,
        verbose=verbose,
    )

    return {
        'results': [result],
        'model': model_name,
        'num_arrays': num_arrays,
        'turbines_per_array': config.turbines_per_array,
        'project_capacity_mw': data['project_capacity_mw'],
        'wake_loss_factor': config.wake_loss_factor,
        'cluster_radius_km': cluster_radius_km,
        'fcr': config.fcr,
        'current_mode': current_mode,
    }


def run_lcoe_optimization(site_data, num_arrays, rated_power_mw,
                           cluster_radius_km=20.0, current_mode="total",
                           config=None, verbose=True):
    """
    Main entry point for min-LCOE optimization.

    Args:
        site_data: Dict from load_site_data_from_npz()
        num_arrays: Number of arrays to deploy
        rated_power_mw: Rated power per turbine (MW)
        cluster_radius_km: Max distance from collection point (km)
        current_mode: "total" or "tidal"
        config: ProjectConfig (defaults to DEFAULT_CONFIG)
        verbose: Print progress

    Returns:
        dict with results list, model name, and project metadata.
    """
    return _run_enumeration_model(
        site_data, num_arrays, rated_power_mw,
        cluster_radius_km, current_mode, config,
        objective="min_lcoe", model_name="min_lcoe", verbose=verbose,
    )


def run_generation_optimization(site_data, num_arrays, rated_power_mw,
                                 cluster_radius_km=20.0, current_mode="total",
                                 config=None, verbose=True):
    """
    Main entry point for max-generation optimization.

    Args:
        site_data: Dict from load_site_data_from_npz()
        num_arrays: Number of arrays to deploy
        rated_power_mw: Rated power per turbine (MW)
        cluster_radius_km: Max distance from collection point (km)
        current_mode: "total" or "tidal"
        config: ProjectConfig (defaults to DEFAULT_CONFIG)
        verbose: Print progress

    Returns:
        dict with results list, model name, and project metadata.
    """
    return _run_enumeration_model(
        site_data, num_arrays, rated_power_mw,
        cluster_radius_km, current_mode, config,
        objective="max_generation", model_name="max_generation", verbose=verbose,
    )
