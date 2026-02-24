"""
Multi-Collection-Point Optimization Model
==========================================

Builds a Pyomo optimization model that jointly selects K collection points
and N arrays in a single solve (facility-location + portfolio optimization).

Variables:
    x_i in {0,1}  — deploy array at site i
    y_j in {0,1}  — open CP at site j  (j in P)
    z_ij in {0,1} — assign array i to CP j  (sparse: only where dist <= R)

Objective: Minimize portfolio variance  (x^T Sigma x)

Constraints:
    1. sum x_i = N               (exactly N arrays)
    2. sum y_j = K               (exactly K CPs)
    3. sum_j z_ij = x_i          (each deployed array assigned to one CP)
    4. z_ij <= y_j               (only assign to open CPs)
    5. LCOE <= target            (linearized cost/energy constraint)
    6. floor(N/K) y_j <= sum_i z_ij <= ceil(N/K) y_j  (balanced load)
    7. x_i <= sum_j y_j          (cover: tightens LP relaxation)

Functions:
    - find_cp_candidates: Pre-filter sites that can serve as CPs
    - build_sparse_arcs: Build (site, CP) pairs within cluster radius
    - precompute_inter_array_costs: Inter-array cable cost per arc
    - precompute_transmission_costs: Transmission cost per candidate CP
    - build_multi_cp_model: Create Pyomo ConcreteModel
    - evaluate_multi_cp_solution: Evaluate metrics for a solution
"""

import math
import numpy as np

from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    Binary,
    Set,
    minimize,
    value,
)

from ..costs.electrical.inter_array import calculate_single_inter_array_cost
from ..costs.electrical.transmission import calculate_transmission_cost


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def find_cp_candidates(distance_matrix, cluster_radius_km, min_sites):
    """
    Find sites that can serve as collection points.

    A site is a candidate CP if it has at least min_sites neighbors
    (including itself) within cluster_radius_km.

    Args:
        distance_matrix: Pairwise distance matrix (n x n) in km
        cluster_radius_km: Maximum distance from CP to served array
        min_sites: Minimum number of reachable sites required

    Returns:
        List of candidate CP indices (sorted)
    """
    n = distance_matrix.shape[0]
    candidates = []
    for j in range(n):
        n_within = int(np.sum(distance_matrix[j] <= cluster_radius_km))
        if n_within >= min_sites:
            candidates.append(j)
    return candidates


def build_sparse_arcs(distance_matrix, cp_candidates, cluster_radius_km):
    """
    Build sparse (site, CP) arc list.

    Only creates arcs where site i is within cluster_radius_km of CP j.

    Args:
        distance_matrix: Pairwise distance matrix (n x n) in km
        cp_candidates: List of candidate CP indices
        cluster_radius_km: Maximum distance for an arc

    Returns:
        List of (i, j) tuples where i is a site index and j is a CP index
    """
    n = distance_matrix.shape[0]
    arcs = []
    cp_set = set(cp_candidates)
    for j in cp_candidates:
        for i in range(n):
            if distance_matrix[i, j] <= cluster_radius_km:
                arcs.append((i, j))
    return arcs


def precompute_inter_array_costs(arcs, distance_matrix, array_power_mw,
                                  inter_array_voltage_v, power_factor, fcr):
    """
    Precompute annualized inter-array cable cost for each arc.

    Args:
        arcs: List of (site, CP) tuples
        distance_matrix: Pairwise distance matrix in km
        array_power_mw: Power per array in MW
        inter_array_voltage_v: Inter-array cable voltage in V
        power_factor: Power factor
        fcr: Fixed charge rate

    Returns:
        Dict {(i, j): annualized_cost} in $/year
    """
    costs = {}
    for (i, j) in arcs:
        d = distance_matrix[i, j]
        result = calculate_single_inter_array_cost(
            d,
            array_power_mw=array_power_mw,
            voltage_v=inter_array_voltage_v,
            power_factor=power_factor,
            fcr=fcr,
        )
        costs[(i, j)] = result['annualized_cost']
    return costs


def precompute_transmission_costs(cp_candidates, dist_to_shore,
                                   cp_capacity_mw, capacity_factor,
                                   fcr, opex_rate):
    """
    Precompute annualized transmission cost for each candidate CP.

    Transmission is sized at cp_capacity_mw (= ceil(N/K) * array_capacity_mw).

    Args:
        cp_candidates: List of candidate CP indices
        dist_to_shore: Array of distances to shore in km (all sites)
        cp_capacity_mw: Transmission capacity to size for each CP
        capacity_factor: Average capacity factor for efficiency calc
        fcr: Fixed charge rate
        opex_rate: OPEX as fraction of CAPEX

    Returns:
        Dict {j: {'annualized_cost': float, 'mode': str}} for each CP
    """
    costs = {}
    for j in cp_candidates:
        shore_dist = dist_to_shore[j]
        trans = calculate_transmission_cost(
            shore_dist, cp_capacity_mw,
            capacity_factor=capacity_factor,
            fcr=fcr,
            opex_rate=opex_rate,
        )
        costs[j] = {
            'annualized_cost': trans['annualized_cost'],
            'mode': trans['mode'],
        }
    return costs


# =========================================================================
# PYOMO MODEL BUILDER
# =========================================================================

def build_multi_cp_model(n_sites, cp_candidates, arcs,
                          num_arrays, num_cps,
                          energy_per_site, covariance_matrix,
                          fixed_cost, inter_costs, trans_costs,
                          lcoe_target):
    """
    Build the multi-CP Pyomo optimization model.

    Args:
        n_sites: Total number of candidate sites
        cp_candidates: List of CP candidate indices
        arcs: List of (site, CP) tuples
        num_arrays: N — number of arrays to deploy
        num_cps: K — number of CPs to open
        energy_per_site: Net annual energy per site (MWh/year), length n_sites
        covariance_matrix: Covariance matrix (n_sites x n_sites)
        fixed_cost: Total fixed cost for the fleet ($/year)
        inter_costs: Dict {(i,j): annualized_cost} for inter-array cables
        trans_costs: Dict {j: {'annualized_cost': float}} for transmission
        lcoe_target: Maximum LCOE ($/MWh)

    Returns:
        Pyomo ConcreteModel ready for solving
    """
    model = ConcreteModel()

    # --- Index sets ---
    model.Sites = Set(initialize=range(n_sites))
    model.CPs = Set(initialize=cp_candidates)
    model.Arcs = Set(initialize=arcs, dimen=2)

    # --- Decision variables ---
    model.x = Var(model.Sites, domain=Binary)   # deploy array at site i
    model.y = Var(model.CPs, domain=Binary)      # open CP at site j
    model.z = Var(model.Arcs, domain=Binary)      # assign site i to CP j

    # --- Store data ---
    model.energy = energy_per_site
    model.cov_matrix = covariance_matrix
    model.fixed_cost = fixed_cost
    model.inter_costs = inter_costs
    model.trans_costs = trans_costs
    model.num_arrays = num_arrays
    model.num_cps = num_cps
    model.lcoe_target = lcoe_target

    # Balance bounds
    model.lb_per_cp = num_arrays // num_cps          # floor(N/K)
    model.ub_per_cp = math.ceil(num_arrays / num_cps)  # ceil(N/K)

    # Precompute: for each site, which CPs can it reach?
    site_to_cps = {}
    for (i, j) in arcs:
        site_to_cps.setdefault(i, []).append(j)
    model.site_to_cps = site_to_cps

    # Precompute: for each CP, which sites can it serve?
    cp_to_sites = {}
    for (i, j) in arcs:
        cp_to_sites.setdefault(j, []).append(i)
    model.cp_to_sites = cp_to_sites

    # -----------------------------------------------------------------
    # Objective: Minimize portfolio variance (x^T Sigma x)
    # -----------------------------------------------------------------
    def variance_rule(m):
        return sum(
            m.x[i] * m.x[k] * m.cov_matrix[i, k]
            for i in m.Sites
            for k in m.Sites
        )

    model.objective = Objective(rule=variance_rule, sense=minimize)

    # -----------------------------------------------------------------
    # Constraint 1: Exactly N arrays
    # -----------------------------------------------------------------
    def array_count_rule(m):
        return sum(m.x[i] for i in m.Sites) == m.num_arrays

    model.array_count = Constraint(rule=array_count_rule)

    # -----------------------------------------------------------------
    # Constraint 2: Exactly K CPs
    # -----------------------------------------------------------------
    def cp_count_rule(m):
        return sum(m.y[j] for j in m.CPs) == m.num_cps

    model.cp_count = Constraint(rule=cp_count_rule)

    # -----------------------------------------------------------------
    # Constraint 3: Each deployed array assigned to exactly one CP
    #   sum_j z_ij = x_i  for all i
    # -----------------------------------------------------------------
    def assignment_rule(m, i):
        cps_for_i = m.site_to_cps.get(i, [])
        if not cps_for_i:
            # Site has no reachable CP — force x_i = 0
            return m.x[i] == 0
        return sum(m.z[i, j] for j in cps_for_i) == m.x[i]

    model.assignment = Constraint(model.Sites, rule=assignment_rule)

    # -----------------------------------------------------------------
    # Constraint 4: Only assign to open CPs
    #   z_ij <= y_j  for all (i,j) in Arcs
    # -----------------------------------------------------------------
    def linking_rule(m, i, j):
        return m.z[i, j] <= m.y[j]

    model.linking = Constraint(model.Arcs, rule=linking_rule)

    # -----------------------------------------------------------------
    # Constraint 5: LCOE constraint (linearized)
    #   fixed + sum inter_ij * z_ij + sum trans_j * y_j
    #       - target * sum energy_i * x_i  <= 0
    # -----------------------------------------------------------------
    def lcoe_rule(m):
        total_inter = sum(
            m.inter_costs[i, j] * m.z[i, j]
            for (i, j) in m.Arcs
        )
        total_trans = sum(
            m.trans_costs[j]['annualized_cost'] * m.y[j]
            for j in m.CPs
        )
        total_energy = sum(m.energy[i] * m.x[i] for i in m.Sites)

        return (
            m.fixed_cost + total_inter + total_trans
            - m.lcoe_target * total_energy
        ) <= 0

    model.lcoe_constraint = Constraint(rule=lcoe_rule)

    # -----------------------------------------------------------------
    # Constraint 6: Balanced load per CP
    #   floor(N/K) * y_j <= sum_i z_ij <= ceil(N/K) * y_j  for j in P
    # -----------------------------------------------------------------
    def balance_lb_rule(m, j):
        sites_for_j = m.cp_to_sites.get(j, [])
        if not sites_for_j:
            return Constraint.Skip
        return m.lb_per_cp * m.y[j] <= sum(m.z[i, j] for i in sites_for_j)

    def balance_ub_rule(m, j):
        sites_for_j = m.cp_to_sites.get(j, [])
        if not sites_for_j:
            return Constraint.Skip
        return sum(m.z[i, j] for i in sites_for_j) <= m.ub_per_cp * m.y[j]

    model.balance_lb = Constraint(model.CPs, rule=balance_lb_rule)
    model.balance_ub = Constraint(model.CPs, rule=balance_ub_rule)

    # -----------------------------------------------------------------
    # Constraint 7: Cover — array needs at least one reachable open CP
    #   x_i <= sum_{j: (i,j) in Arcs} y_j   for all i
    # -----------------------------------------------------------------
    def cover_rule(m, i):
        cps_for_i = m.site_to_cps.get(i, [])
        if not cps_for_i:
            return Constraint.Skip
        return m.x[i] <= sum(m.y[j] for j in cps_for_i)

    model.cover = Constraint(model.Sites, rule=cover_rule)

    return model


# =========================================================================
# SOLUTION EVALUATION
# =========================================================================

def evaluate_multi_cp_solution(x_vals, y_vals, z_vals,
                                energy_per_site, covariance_matrix,
                                fixed_cost, inter_costs, trans_costs):
    """
    Evaluate metrics for a multi-CP solution.

    Args:
        x_vals: Dict {i: 0/1} or array of site deployment decisions
        y_vals: Dict {j: 0/1} of CP opening decisions
        z_vals: Dict {(i,j): 0/1} of assignment decisions
        energy_per_site: Net annual energy per site (MWh/year)
        covariance_matrix: Covariance matrix (n x n)
        fixed_cost: Total fixed cost ($/year)
        inter_costs: Dict {(i,j): annualized_cost}
        trans_costs: Dict {j: {'annualized_cost': float, 'mode': str}}

    Returns:
        dict with variance, lcoe, cost breakdown, assignments, etc.
    """
    # Selected sites and CPs
    selected_sites = sorted(i for i, v in x_vals.items() if round(v) == 1)
    active_cps = sorted(j for j, v in y_vals.items() if round(v) == 1)

    # Build assignment map: {cp: [sites]}
    assignments = {j: [] for j in active_cps}
    for (i, j), v in z_vals.items():
        if round(v) == 1:
            assignments.setdefault(j, []).append(i)
    for j in assignments:
        assignments[j].sort()

    # Variance
    n = len(energy_per_site)
    x_vec = np.zeros(n)
    for i in selected_sites:
        x_vec[i] = 1.0
    variance = float(x_vec @ covariance_matrix @ x_vec)

    # Costs
    cost_inter = sum(
        inter_costs[i, j]
        for (i, j), v in z_vals.items()
        if round(v) == 1
    )
    cost_trans = sum(
        trans_costs[j]['annualized_cost']
        for j in active_cps
    )
    total_cost = fixed_cost + cost_inter + cost_trans

    # Energy
    total_energy = float(np.sum(energy_per_site[selected_sites]))

    # LCOE
    lcoe = total_cost / total_energy if total_energy > 0 else float('inf')

    # Per-CP details
    per_cp = []
    for j in active_cps:
        per_cp.append({
            'cp_index': j,
            'num_arrays_served': len(assignments[j]),
            'assigned_sites': assignments[j],
            'transmission_cost': trans_costs[j]['annualized_cost'],
            'transmission_mode': trans_costs[j]['mode'],
            'inter_array_cost': sum(
                inter_costs[i, j] for i in assignments[j]
                if (i, j) in inter_costs
            ),
        })

    return {
        'selected_sites': selected_sites,
        'active_cps': active_cps,
        'assignments': assignments,
        'variance': variance,
        'achieved_lcoe': lcoe,
        'total_cost': total_cost,
        'total_energy': total_energy,
        'cost_fixed': fixed_cost,
        'cost_inter_array': cost_inter,
        'cost_transmission': cost_trans,
        'per_cp_details': per_cp,
    }
