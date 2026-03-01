"""
Enumeration Solver
==================

Brute-force enumeration of all C(n, k) site subsets.

With num_arrays=3 and ~10-50 candidates per collection point,
C(n, 3) <= 20K combinations -- trivially fast without Pyomo/Gurobi.

Used by lcoe_model and generation_model.
"""

import itertools
import numpy as np

from ..costs import calculate_inter_array_cost, calculate_transmission_cost


def find_best_subset(n_candidates, num_arrays, energy_per_site,
                     covariance_matrix, total_fixed_cost,
                     inter_array_costs, transmission_cost,
                     objective="min_lcoe"):
    """
    Enumerate all C(n, k) subsets, return only the best by objective.

    Args:
        n_candidates: Number of candidate sites
        num_arrays: Number of arrays to select (k)
        energy_per_site: Array of net annual energy per site (MWh/year)
        covariance_matrix: n x n covariance matrix
        total_fixed_cost: Fixed cost for the fleet ($/year)
        inter_array_costs: Per-site inter-array cable cost ($/year)
        transmission_cost: Export cable cost ($/year)
        objective: "min_lcoe" or "max_generation"

    Returns:
        Best result dict {indices, lcoe, variance, total_cost, total_energy,
        cost_breakdown}, or None if no valid subset exists.
    """
    best = None
    best_val = float('inf')

    for combo in itertools.combinations(range(n_candidates), num_arrays):
        indices = np.array(combo)

        total_energy = float(np.sum(energy_per_site[indices]))
        if total_energy <= 0:
            continue

        inter_array_total = float(np.sum(inter_array_costs[indices]))
        total_cost = total_fixed_cost + inter_array_total + transmission_cost
        lcoe = total_cost / total_energy

        # Check if this is the new best
        val = lcoe if objective == "min_lcoe" else -total_energy
        if val >= best_val:
            continue
        best_val = val

        cov_sub = covariance_matrix[np.ix_(indices, indices)]
        variance = float(np.sum(cov_sub))

        best = {
            'indices': indices,
            'lcoe': lcoe,
            'variance': variance,
            'total_cost': total_cost,
            'total_energy': total_energy,
            'cost_breakdown': {
                'fixed': total_fixed_cost,
                'inter_array': inter_array_total,
                'transmission': transmission_cost,
            },
        }

    return best


def prepare_cluster_inputs(cp_idx, distance_matrix, cluster_radius_km,
                           energy_vector, covariance_matrix, site_data,
                           project_capacity_mw, array_power_mw, config):
    """
    Compute per-cluster data for a single collection point.

    Returns:
        (candidates, energy_subset, cov_subset, inter_array_costs,
         transmission_cost, transmission_mode, shore_dist)
        or None if too few candidates.
    """
    candidates = np.where(distance_matrix[cp_idx] <= cluster_radius_km)[0]

    distances_to_cp = distance_matrix[cp_idx, candidates]
    inter_array_costs = np.array([
        calculate_inter_array_cost(
            d, array_power_mw=array_power_mw, config=config,
        )['annualized_cost']
        for d in distances_to_cp
    ])

    shore_dist = site_data['dist_to_shore_km'][cp_idx]
    cp_capacity_factor = float(site_data['capacity_factors'][cp_idx])
    trans = calculate_transmission_cost(
        shore_dist, project_capacity_mw,
        capacity_factor=cp_capacity_factor, config=config,
    )

    energy_subset = energy_vector[candidates]
    cov_subset = covariance_matrix[np.ix_(candidates, candidates)]

    return (candidates, energy_subset, cov_subset, inter_array_costs,
            trans['annualized_cost'], trans['mode'], shore_dist)
