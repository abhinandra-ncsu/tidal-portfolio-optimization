"""
Optimization Model Definition
=============================

Builds the Pyomo optimization model for array placement.

Objective: Minimize portfolio variance
Subject to:
    1. Deploy exactly N arrays
    2. LCOE <= target

Functions:
    - validate_model_inputs: Validate input arrays and dimensions
    - build_optimization_model: Create Pyomo ConcreteModel
    - get_model_solution: Extract decision variable values
    - evaluate_solution: Evaluate metrics for a binary solution vector
"""

import numpy as np

from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    Binary,
    RangeSet,
    minimize,
    value,
)


def validate_model_inputs(n_candidates, num_arrays, energy_per_site,
                           covariance_matrix, inter_array_costs):
    """
    Validate optimization model inputs.

    Args:
        n_candidates: Number of candidate sites
        num_arrays: Number of arrays to deploy
        energy_per_site: Net annual energy per site (MWh/year), length n_candidates
        covariance_matrix: Covariance matrix (n_candidates x n_candidates)
        inter_array_costs: Inter-array cable cost per site ($/year), length n_candidates

    Raises:
        ValueError: If inputs are inconsistent
    """
    if n_candidates < num_arrays:
        raise ValueError(
            f"Not enough candidates ({n_candidates}) "
            f"for {num_arrays} arrays"
        )
    if len(energy_per_site) != n_candidates:
        raise ValueError("energy_per_site length mismatch")
    if len(inter_array_costs) != n_candidates:
        raise ValueError("inter_array_costs length mismatch")
    if covariance_matrix.shape != (n_candidates, n_candidates):
        raise ValueError("covariance_matrix shape mismatch")


def build_optimization_model(n_candidates, num_arrays, energy_per_site,
                              covariance_matrix, fixed_cost_per_array,
                              inter_array_costs, transmission_cost,
                              lcoe_target):
    """
    Build the Pyomo optimization model.

    Decision Variables:
        x[i] in {0, 1} - whether to place array at site i

    Objective:
        Minimize: sum_i sum_j (x_i * x_j * Cov_ij)  (portfolio variance)

    Constraints:
        1. sum_i x_i = N  (deploy exactly N arrays)
        2. Total Cost / Total Energy <= LCOE_target
           (rearranged to linear form)

    Args:
        n_candidates: Number of candidate sites
        num_arrays: Number of arrays to deploy
        energy_per_site: Net annual energy per site (MWh/year)
        covariance_matrix: Covariance matrix (n x n)
        fixed_cost_per_array: Device + intra-array cost ($/year)
        inter_array_costs: Inter-array cable cost per site ($/year)
        transmission_cost: Export cable cost ($/year)
        lcoe_target: Maximum LCOE ($/MWh)

    Returns:
        Pyomo ConcreteModel ready for solving
    """
    validate_model_inputs(
        n_candidates, num_arrays, energy_per_site,
        covariance_matrix, inter_array_costs,
    )

    model = ConcreteModel()

    # Index set for candidate sites
    model.Sites = RangeSet(0, n_candidates - 1)

    # Decision variables: binary (0 or 1 array at each site)
    model.x = Var(model.Sites, domain=Binary)

    # Store data as model attributes for constraint access
    model.energy = energy_per_site
    model.cov_matrix = covariance_matrix
    model.inter_array_costs = inter_array_costs
    model.fixed_cost = fixed_cost_per_array
    model.transmission_cost = transmission_cost
    model.num_arrays = num_arrays
    model.lcoe_target = lcoe_target

    # -----------------------------------------------------------------
    # Objective: Minimize variance (x^T * Sigma * x)
    # -----------------------------------------------------------------
    def variance_rule(m):
        return sum(
            m.x[i] * m.x[j] * m.cov_matrix[i, j]
            for i in m.Sites
            for j in m.Sites
        )

    model.objective = Objective(rule=variance_rule, sense=minimize)

    # -----------------------------------------------------------------
    # Constraint 1: Deploy exactly num_arrays
    # -----------------------------------------------------------------
    def array_count_rule(m):
        return sum(m.x[i] for i in m.Sites) == m.num_arrays

    model.array_count = Constraint(rule=array_count_rule)

    # -----------------------------------------------------------------
    # Constraint 2: LCOE <= target
    # -----------------------------------------------------------------
    # Total Cost = N * fixed + sum(inter_array * x) + transmission
    # Total Energy = sum(energy * x)
    # LCOE = Total Cost / Total Energy <= target
    #
    # Rearranged (linear form):
    # N * fixed + sum(inter_array * x) + transmission - target * sum(energy * x) <= 0

    def lcoe_rule(m):
        total_fixed = m.num_arrays * m.fixed_cost
        total_inter_array = sum(m.inter_array_costs[i] * m.x[i] for i in m.Sites)
        total_energy = sum(m.energy[i] * m.x[i] for i in m.Sites)

        # Cost - target * energy <= 0
        return (
            total_fixed + total_inter_array + m.transmission_cost
            - m.lcoe_target * total_energy
        ) <= 0

    model.lcoe_constraint = Constraint(rule=lcoe_rule)

    return model


def get_model_solution(pyomo_model):
    """
    Extract solution values from a solved Pyomo model.

    Args:
        pyomo_model: Solved Pyomo ConcreteModel (from build_optimization_model)

    Returns:
        numpy array of x values (continuous or binary)
    """
    return np.array([value(pyomo_model.x[i]) for i in pyomo_model.Sites])


def get_model_objective_value(pyomo_model):
    """
    Get objective (variance) value from a solved Pyomo model.

    Args:
        pyomo_model: Solved Pyomo ConcreteModel

    Returns:
        float: Variance value
    """
    return value(pyomo_model.objective)


def evaluate_solution(x, n_candidates, num_arrays, energy_per_site,
                       covariance_matrix, fixed_cost_per_array,
                       inter_array_costs, transmission_cost):
    """
    Evaluate metrics for a given binary solution vector.

    Args:
        x: Binary solution vector (length n_candidates)
        n_candidates: Number of candidate sites
        num_arrays: Number of arrays deployed
        energy_per_site: Net annual energy per site (MWh/year)
        covariance_matrix: Covariance matrix (n x n)
        fixed_cost_per_array: Device + intra-array cost ($/year)
        inter_array_costs: Inter-array cable cost per site ($/year)
        transmission_cost: Export cable cost ($/year)

    Returns:
        dict with: variance, lcoe, total_cost, total_energy,
                   cost_fixed, cost_inter_array, cost_transmission
    """
    # Variance
    variance = float(x @ covariance_matrix @ x)

    # Costs
    total_fixed = num_arrays * fixed_cost_per_array
    total_inter_array = float(np.sum(inter_array_costs * x))
    total_cost = total_fixed + total_inter_array + transmission_cost

    # Energy
    total_energy = float(np.sum(energy_per_site * x))

    # LCOE
    lcoe = total_cost / total_energy if total_energy > 0 else float('inf')

    return {
        'variance': variance,
        'lcoe': lcoe,
        'total_cost': total_cost,
        'total_energy': total_energy,
        'cost_fixed': total_fixed,
        'cost_inter_array': total_inter_array,
        'cost_transmission': transmission_cost,
    }
