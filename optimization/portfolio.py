"""
Portfolio Optimization Functions
================================

Functions for tidal portfolio optimization.

Iterates over collection points and LCOE targets to find
the optimal array placement that minimizes variance.
"""

import numpy as np

from ..costs import calculate_total_fixed_cost
from ..costs import calculate_inter_array_cost, calculate_transmission_cost
from ..energy import calculate_energy_vector, calculate_covariance
from .solver import solve_optimization_model


def calculate_distance_matrix(latitudes, longitudes):
    """
    Calculate pairwise Haversine distances in km.

    Args:
        latitudes: Array of latitudes (degrees)
        longitudes: Array of longitudes (degrees)

    Returns:
        Distance matrix (n_sites, n_sites) in km
    """
    n = len(latitudes)
    distances = np.zeros((n, n))

    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    R = 6371.0  # Earth radius in km

    for i in range(n):
        for j in range(i + 1, n):
            dlat = lat_rad[j] - lat_rad[i]
            dlon = lon_rad[j] - lon_rad[i]

            a = (np.sin(dlat/2)**2 +
                 np.cos(lat_rad[i]) * np.cos(lat_rad[j]) * np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a))

            distances[i, j] = R * c
            distances[j, i] = distances[i, j]

    return distances


def find_viable_collection_points(distance_matrix, cluster_radius_km, min_sites):
    """
    Find sites that can serve as collection points.

    A site is viable if it has at least min_sites within cluster_radius_km.

    Args:
        distance_matrix: Pairwise distance matrix
        cluster_radius_km: Maximum distance from collection point
        min_sites: Minimum number of sites required

    Returns:
        List of viable collection point indices
    """
    n = distance_matrix.shape[0]
    viable = []

    for i in range(n):
        n_within_radius = np.sum(distance_matrix[i] <= cluster_radius_km)
        if n_within_radius >= min_sites:
            viable.append(i)

    return viable


def optimize_for_lcoe_target(site_data, energy_vector, covariance_matrix,
                              distance_matrix, viable_cps, num_arrays,
                              lcoe_target, cluster_radius_km,
                              total_fixed_cost, project_capacity_mw,
                              array_power_mw, config, verbose=True):
    """
    Optimize for a single LCOE target.

    Iterates over collection points to find the minimum variance solution.

    Args:
        site_data: Dict with site arrays (latitudes, longitudes, dist_to_shore, etc.)
        energy_vector: Net annual energy per site (MWh/year)
        covariance_matrix: Full covariance matrix (n_sites x n_sites)
        distance_matrix: Pairwise distance matrix
        viable_cps: List of viable collection point indices
        num_arrays: Number of arrays to deploy
        lcoe_target: Target LCOE ($/MWh)
        cluster_radius_km: Maximum distance from collection point
        total_fixed_cost: Total fixed cost for the fleet ($/year)
        project_capacity_mw: Total project capacity (MW)
        array_power_mw: Power per array (MW)
        config: ProjectConfig with inter_array_voltage_v, power_factor, fcr, opex_rate
        verbose: Print progress

    Returns:
        dict with optimization result:
            - lcoe_target, feasible
            - selected_sites, collection_point, variance, lcoe
            - total_cost, total_energy, cost_breakdown
            - transmission_mode, dist_to_shore_km
            - n_feasible, n_tried
            - solve_time, solver_used
    """
    best_result = None
    best_variance = float('inf')
    best_cp = None
    best_selected = None

    n_feasible = 0
    n_tried = 0

    for cp_idx in viable_cps:
        # Get candidates within cluster radius
        candidates = np.where(distance_matrix[cp_idx] <= cluster_radius_km)[0]
        n_tried += 1

        # Get energy and covariance for candidates
        energy_subset = energy_vector[candidates]
        cov_subset = covariance_matrix[np.ix_(candidates, candidates)]

        # Inter-array cable costs
        distances_to_cp = distance_matrix[cp_idx, candidates]
        inter_array_costs = np.array([
            calculate_inter_array_cost(
                d, array_power_mw=array_power_mw, config=config,
            )['annualized_cost']
            for d in distances_to_cp
        ])

        # Transmission cost (use collection point's capacity factor)
        shore_dist = site_data['dist_to_shore_km'][cp_idx]
        cp_capacity_factor = float(site_data['capacity_factors'][cp_idx])
        trans = calculate_transmission_cost(
            shore_dist, project_capacity_mw,
            capacity_factor=cp_capacity_factor, config=config,
        )

        # Build model inputs dict
        model_inputs = {
            'n_candidates': len(candidates),
            'num_arrays': num_arrays,
            'energy_per_site': energy_subset,
            'covariance_matrix': cov_subset,
            'total_fixed_cost': total_fixed_cost,
            'inter_array_costs': inter_array_costs,
            'transmission_cost': trans['annualized_cost'],
            'lcoe_target': lcoe_target,
        }

        # Solve
        result = solve_optimization_model(model_inputs, verbose=False)

        if result is not None:
            n_feasible += 1

            if result['variance'] < best_variance:
                best_variance = result['variance']
                best_result = result
                best_cp = cp_idx
                # Map local indices back to global
                best_selected = candidates[result['selected_indices']]

    # Build result
    if best_result is not None:
        if verbose:
            print(f"  ✓ Feasible: {n_feasible}/{n_tried} collection points")
            print(f"    Best variance: {best_variance:.2f} MW²")
            print(f"    LCOE achieved: ${best_result['lcoe']:.0f}/MWh")

        # Get transmission mode
        shore_dist = site_data['dist_to_shore_km'][best_cp]
        cp_capacity_factor = float(site_data['capacity_factors'][best_cp])
        trans = calculate_transmission_cost(
            shore_dist, project_capacity_mw,
            capacity_factor=cp_capacity_factor, config=config,
        )

        return {
            'lcoe_target': lcoe_target,
            'feasible': True,
            'selected_sites': best_selected,
            'collection_point': best_cp,
            'variance': best_result['variance'],
            'lcoe': best_result['lcoe'],
            'total_cost': best_result['total_cost'],
            'total_energy': best_result['total_energy'],
            'cost_breakdown': best_result['cost_breakdown'],
            'transmission_mode': trans['mode'],
            'dist_to_shore_km': shore_dist,
            'n_feasible': n_feasible,
            'n_tried': n_tried,
            'solve_time': best_result['solve_time'],
            'solver_used': best_result['solver_used'],
        }
    else:
        if verbose:
            print(f"  ✗ Infeasible (tried {n_tried} collection points)")

        return {
            'lcoe_target': lcoe_target,
            'feasible': False,
            'n_feasible': 0,
            'n_tried': n_tried,
        }


def run_portfolio_optimization(site_data, num_arrays, lcoe_targets,
                                rated_power_mw, cluster_radius_km=20.0,
                                current_mode="total",
                                config=None,
                                verbose=True):
    """
    Run portfolio optimization.

    Main entry point for portfolio optimization. Engineering constants are
    bundled in a ``ProjectConfig``; pass ``config=ProjectConfig(...)`` to
    override defaults.

    Args:
        site_data: Dict from load_site_data_from_npz() with:
            - latitudes, longitudes: Coordinates
            - capacity_factors: 0-1 scale
            - dist_to_shore: km
            - power_timeseries: (n_sites, n_timesteps) MW
            When current_mode="tidal", also needs:
            - tidal_capacity_factors: 0-1 scale (tidal-only)
            - tidal_power_timeseries: (n_sites, n_timesteps) MW (tidal-only)

        num_arrays: Number of arrays to deploy
        lcoe_targets: List of LCOE targets to test ($/MWh)
        rated_power_mw: Rated power per turbine (MW)
        cluster_radius_km: Max distance from collection point (km). Default 20.
        current_mode: "total" (default) or "tidal" -- selects which timeseries
            to use for energy/covariance calculations
        config: ProjectConfig with all engineering constants.
            Defaults to DEFAULT_CONFIG.
        verbose: Print progress

    Returns:
        dict with:
            - results: List of result dicts for each LCOE target
            - num_arrays, turbines_per_array, project_capacity_mw
            - wake_loss_factor, cluster_radius_km, fcr, current_mode
            - site_data: Original site data
    """
    from ..config import DEFAULT_CONFIG
    if config is None:
        config = DEFAULT_CONFIG

    # Sort LCOE targets
    lcoe_targets = sorted(lcoe_targets)

    # Project capacity
    array_power_mw = config.turbines_per_array * rated_power_mw
    project_capacity_mw = num_arrays * array_power_mw

    # Calculate distance matrix
    if verbose:
        print("Calculating distance matrix...")
    distance_matrix = calculate_distance_matrix(
        site_data['latitudes'],
        site_data['longitudes'],
    )

    # Select timeseries based on current_mode
    if current_mode == "tidal":
        if 'tidal_capacity_factors' not in site_data or 'tidal_power_timeseries' not in site_data:
            raise ValueError(
                "current_mode='tidal' requires 'tidal_capacity_factors' and "
                "'tidal_power_timeseries' in site_data. Run the UTide pipeline first."
            )
        active_cf = site_data['tidal_capacity_factors']
        active_power = site_data['tidal_power_timeseries']
    else:
        active_cf = site_data['capacity_factors']
        active_power = site_data['power_timeseries']

    # Calculate energy vector and covariance matrix
    if verbose:
        mode_label = "tidal-only" if current_mode == "tidal" else "total"
        print(f"Preparing energy data (current_mode={mode_label})...")
    energy_vector = calculate_energy_vector(active_cf, rated_power_mw, config)
    cov_result = calculate_covariance(active_power, config, scaled=True)
    covariance_matrix = cov_result['covariance_matrix']

    # Find viable collection points
    if verbose:
        print("Finding viable collection points...")
    viable_cps = find_viable_collection_points(
        distance_matrix, cluster_radius_km, num_arrays
    )
    if verbose:
        print(f"  Found {len(viable_cps)} viable collection points")

    if len(viable_cps) == 0:
        raise ValueError(
            f"No viable collection points found with radius {cluster_radius_km} km"
        )

    # Calculate total fixed cost for the fleet (economies of scale)
    total_fixed_cost = calculate_total_fixed_cost(
        num_arrays=num_arrays,
        turbine_power_mw=rated_power_mw,
        config=config,
    )

    # Run optimization for each LCOE target
    results = []

    for lcoe_idx, lcoe_target in enumerate(lcoe_targets):
        if verbose:
            print(f"\n[{lcoe_idx+1}/{len(lcoe_targets)}] LCOE target: ${lcoe_target:.0f}/MWh")

        result = optimize_for_lcoe_target(
            site_data=site_data,
            energy_vector=energy_vector,
            covariance_matrix=covariance_matrix,
            distance_matrix=distance_matrix,
            viable_cps=viable_cps,
            num_arrays=num_arrays,
            lcoe_target=lcoe_target,
            cluster_radius_km=cluster_radius_km,
            total_fixed_cost=total_fixed_cost,
            project_capacity_mw=project_capacity_mw,
            array_power_mw=array_power_mw,
            config=config,
            verbose=verbose,
        )
        results.append(result)

    return {
        'results': results,
        'num_arrays': num_arrays,
        'turbines_per_array': config.turbines_per_array,
        'project_capacity_mw': project_capacity_mw,
        'wake_loss_factor': config.wake_loss_factor,
        'cluster_radius_km': cluster_radius_km,
        'fcr': config.fcr,
        'current_mode': current_mode,
        'lcoe_targets': np.array(lcoe_targets),
        'site_data': site_data,
    }


def get_best_result(results):
    """
    Get result with minimum feasible LCOE.

    Args:
        results: List of result dicts (from run_portfolio_optimization)

    Returns:
        Best result dict or None if none feasible
    """
    feasible = [r for r in results if r['feasible']]
    if not feasible:
        return None
    return min(feasible, key=lambda r: r['lcoe'])
