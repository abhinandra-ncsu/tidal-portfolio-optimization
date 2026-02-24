"""
Multi-Collection-Point Portfolio Optimization
==============================================

Orchestration module for the multi-CP optimizer. Jointly selects K collection
points and N arrays in a single global solve per LCOE target.

Main entry point:
    run_multi_cp_optimization(site_data, num_arrays, num_cps, lcoe_targets, ...)

Flow:
    1. Calculate distance matrix
    2. Prepare energy data (energy vector + covariance)
    3. Find CP candidates + build sparse arcs
    4. Precompute inter-array and transmission costs
    5. Calculate fixed cost
    6. For each LCOE target: build model -> solve -> collect result
    7. Return results dict
"""

import math
import numpy as np

from ..costs import calculate_total_fixed_cost
from ..energy import prepare_energy_data
from .portfolio import calculate_distance_matrix
from .multi_cp_model import (
    find_cp_candidates,
    build_sparse_arcs,
    precompute_inter_array_costs,
    precompute_transmission_costs,
)
from .multi_cp_solver import solve_multi_cp_model


def run_multi_cp_optimization(site_data, num_arrays, num_cps, lcoe_targets,
                               wake_loss_factor, turbines_per_array,
                               rated_power_mw, cluster_radius_km,
                               fcr, rows, cols, power_factor,
                               row_spacing_km, col_spacing_km,
                               intra_array_voltage_v, inter_array_voltage_v,
                               capacity_factor, opex_rate,
                               current_mode="total",
                               timeout=600, verbose=True):
    """
    Run multi-CP portfolio optimization.

    Jointly selects K collection points and N arrays in a single solve,
    minimizing portfolio variance subject to LCOE and balance constraints.

    Args:
        site_data: Dict from load_site_data_from_npz() with:
            - latitudes, longitudes, capacity_factors, dist_to_shore
            - power_timeseries: (n_sites, n_timesteps) MW
            When current_mode="tidal", also needs tidal_* variants.
        num_arrays: N — number of arrays to deploy
        num_cps: K — number of collection points to open
        lcoe_targets: List of LCOE targets to test ($/MWh)
        wake_loss_factor: Wake loss multiplier (e.g., 0.88)
        turbines_per_array: Turbines per array
        rated_power_mw: Rated power per turbine (MW)
        cluster_radius_km: Max distance from CP to served array (km)
        fcr: Fixed charge rate
        rows, cols: Array layout dimensions
        power_factor: AC power factor
        row_spacing_km, col_spacing_km: Array spacing
        intra_array_voltage_v: Intra-array cable voltage (V)
        inter_array_voltage_v: Inter-array cable voltage (V)
        capacity_factor: Average capacity factor (for transmission sizing)
        opex_rate: OPEX as fraction of CAPEX
        current_mode: "total" or "tidal"
        timeout: Solver timeout per LCOE target (seconds)
        verbose: Print progress

    Returns:
        dict with:
            - results: List of result dicts per LCOE target
            - num_arrays, num_cps, turbines_per_array, project_capacity_mw
            - wake_loss_factor, cluster_radius_km, fcr, current_mode
            - n_cp_candidates, n_arcs
            - site_data: Original site data
    """
    lcoe_targets = sorted(lcoe_targets)

    # Project sizing
    array_power_mw = turbines_per_array * rated_power_mw
    project_capacity_mw = num_arrays * array_power_mw
    cp_capacity_mw = math.ceil(num_arrays / num_cps) * array_power_mw
    min_sites_per_cp = math.ceil(num_arrays / num_cps)

    # -------------------------------------------------------------------------
    # Step 1: Distance matrix
    # -------------------------------------------------------------------------
    if verbose:
        print("Calculating distance matrix...")
    distance_matrix = calculate_distance_matrix(
        site_data['latitudes'],
        site_data['longitudes'],
    )
    n_sites = len(site_data['latitudes'])

    # -------------------------------------------------------------------------
    # Step 2: Energy data
    # -------------------------------------------------------------------------
    if current_mode == "tidal":
        if 'tidal_capacity_factors' not in site_data:
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

    energy_data = prepare_energy_data(
        capacity_factors=active_cf,
        power_timeseries=active_power,
        rated_power_mw=rated_power_mw,
        turbines_per_array=turbines_per_array,
        wake_loss_factor=wake_loss_factor,
    )
    energy_vector = energy_data['energy_vector']
    covariance_matrix = energy_data['covariance_matrix']

    # -------------------------------------------------------------------------
    # Step 3: CP candidates and sparse arcs
    # -------------------------------------------------------------------------
    if verbose:
        print("Finding CP candidates...")
    cp_candidates = find_cp_candidates(
        distance_matrix, cluster_radius_km, min_sites_per_cp,
    )
    if verbose:
        print(f"  Found {len(cp_candidates)} CP candidates")

    if len(cp_candidates) < num_cps:
        raise ValueError(
            f"Only {len(cp_candidates)} CP candidates found "
            f"(need {num_cps}). Try increasing cluster_radius_km."
        )

    if verbose:
        print("Building sparse arcs...")
    arcs = build_sparse_arcs(distance_matrix, cp_candidates, cluster_radius_km)
    if verbose:
        print(f"  Built {len(arcs)} arcs")

    # -------------------------------------------------------------------------
    # Step 4: Precompute costs
    # -------------------------------------------------------------------------
    if verbose:
        print("Precomputing inter-array costs...")
    inter_costs = precompute_inter_array_costs(
        arcs, distance_matrix,
        array_power_mw=array_power_mw,
        inter_array_voltage_v=inter_array_voltage_v,
        power_factor=power_factor,
        fcr=fcr,
    )

    if verbose:
        print("Precomputing transmission costs...")
    trans_costs = precompute_transmission_costs(
        cp_candidates,
        dist_to_shore=site_data['dist_to_shore'],
        cp_capacity_mw=cp_capacity_mw,
        capacity_factor=capacity_factor,
        fcr=fcr,
        opex_rate=opex_rate,
    )

    # -------------------------------------------------------------------------
    # Step 5: Fixed cost (device + intra-array)
    # -------------------------------------------------------------------------
    total_fixed_cost = calculate_total_fixed_cost(
        num_arrays=num_arrays,
        turbines_per_array=turbines_per_array,
        fcr=fcr,
        rows=rows,
        cols=cols,
        turbine_power_mw=rated_power_mw,
        power_factor=power_factor,
        row_spacing_km=row_spacing_km,
        col_spacing_km=col_spacing_km,
        intra_array_voltage_v=intra_array_voltage_v,
    )

    # -------------------------------------------------------------------------
    # Step 6: Solve for each LCOE target
    # -------------------------------------------------------------------------
    results = []

    for lcoe_idx, lcoe_target in enumerate(lcoe_targets):
        if verbose:
            print(f"\n[{lcoe_idx+1}/{len(lcoe_targets)}] "
                  f"LCOE target: ${lcoe_target:.0f}/MWh")

        model_inputs = {
            'n_sites': n_sites,
            'cp_candidates': cp_candidates,
            'arcs': arcs,
            'num_arrays': num_arrays,
            'num_cps': num_cps,
            'energy_per_site': energy_vector,
            'covariance_matrix': covariance_matrix,
            'fixed_cost': total_fixed_cost,
            'inter_costs': inter_costs,
            'trans_costs': trans_costs,
            'lcoe_target': lcoe_target,
        }

        result = solve_multi_cp_model(
            model_inputs, timeout=timeout, verbose=False,
        )

        if result is not None:
            if verbose:
                print(f"  Feasible: variance={result['variance']:.2f} MW^2, "
                      f"LCOE=${result['achieved_lcoe']:.0f}/MWh")
                for cp_detail in result['per_cp_details']:
                    print(f"    CP {cp_detail['cp_index']}: "
                          f"{cp_detail['num_arrays_served']} arrays, "
                          f"{cp_detail['transmission_mode']}")

            results.append({
                'lcoe_target': lcoe_target,
                'feasible': True,
                'selected_sites': result['selected_sites'],
                'active_cps': result['active_cps'],
                'assignments': result['assignments'],
                'variance': result['variance'],
                'achieved_lcoe': result['achieved_lcoe'],
                'cost_fixed': result['cost_fixed'],
                'cost_inter_array': result['cost_inter_array'],
                'cost_transmission': result['cost_transmission'],
                'cost_total': result['total_cost'],
                'total_energy': result['total_energy'],
                'per_cp_details': result['per_cp_details'],
                'solve_time': result['solve_time'],
                'solver_used': result['solver_used'],
            })
        else:
            if verbose:
                print(f"  Infeasible")
            results.append({
                'lcoe_target': lcoe_target,
                'feasible': False,
            })

    return {
        'results': results,
        'num_arrays': num_arrays,
        'num_cps': num_cps,
        'turbines_per_array': turbines_per_array,
        'project_capacity_mw': project_capacity_mw,
        'wake_loss_factor': wake_loss_factor,
        'cluster_radius_km': cluster_radius_km,
        'fcr': fcr,
        'current_mode': current_mode,
        'lcoe_targets': np.array(lcoe_targets),
        'n_cp_candidates': len(cp_candidates),
        'n_arcs': len(arcs),
        'site_data': site_data,
    }


def get_best_multi_cp_result(results):
    """
    Get result with minimum feasible LCOE from multi-CP results.

    Args:
        results: List of result dicts (from run_multi_cp_optimization)

    Returns:
        Best result dict or None if none feasible
    """
    feasible = [r for r in results if r['feasible']]
    if not feasible:
        return None
    return min(feasible, key=lambda r: r['achieved_lcoe'])
