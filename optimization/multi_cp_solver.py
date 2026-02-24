"""
Multi-Collection-Point Solver
==============================

Solves the multi-CP binary quadratic program using Gurobi.

Functions:
    - solve_multi_cp_model: Build + solve the multi-CP optimization model
"""

import time
import numpy as np

from pyomo.environ import (
    SolverFactory,
    SolverStatus,
    TerminationCondition,
    value,
)

from .multi_cp_model import build_multi_cp_model, evaluate_multi_cp_solution


def solve_multi_cp_model(model_inputs, timeout=600, verbose=False):
    """
    Build and solve the multi-CP optimization model using Gurobi.

    Args:
        model_inputs: Dict with all model inputs.
            Required keys: n_sites, cp_candidates, arcs, num_arrays, num_cps,
            energy_per_site, covariance_matrix, fixed_cost, inter_costs,
            trans_costs, lcoe_target
        timeout: Solver time limit in seconds (default: 600)
        verbose: Print solver output

    Returns:
        dict with solver result if successful, None if infeasible.
        Keys: success, selected_sites, active_cps, assignments,
              variance, achieved_lcoe, cost breakdown, per_cp_details,
              solve_time, solver_used, message
    """
    start_time = time.time()

    # Build Pyomo model
    pyomo_model = build_multi_cp_model(
        n_sites=model_inputs['n_sites'],
        cp_candidates=model_inputs['cp_candidates'],
        arcs=model_inputs['arcs'],
        num_arrays=model_inputs['num_arrays'],
        num_cps=model_inputs['num_cps'],
        energy_per_site=model_inputs['energy_per_site'],
        covariance_matrix=model_inputs['covariance_matrix'],
        fixed_cost=model_inputs['fixed_cost'],
        inter_costs=model_inputs['inter_costs'],
        trans_costs=model_inputs['trans_costs'],
        lcoe_target=model_inputs['lcoe_target'],
    )

    # Configure Gurobi
    solver = SolverFactory('gurobi')
    if solver is None or not solver.available():
        raise RuntimeError("Gurobi solver not available")

    solver.options['TimeLimit'] = timeout
    solver.options['MIPGap'] = 0.02        # 2% optimality gap (larger problem)
    solver.options['MIPFocus'] = 1         # Focus on finding feasible solutions
    solver.options['Threads'] = 0          # Use all available threads
    solver.options['OutputFlag'] = 1 if verbose else 0

    # Solve
    results = solver.solve(pyomo_model, tee=verbose)

    # Check solver status
    status = results.solver.status
    termination = results.solver.termination_condition

    # Accept optimal/feasible solutions, and also time-limit incumbents
    ok_status = status == SolverStatus.ok
    aborted_with_solution = (
        status == SolverStatus.aborted
        and termination == TerminationCondition.maxTimeLimit
    )
    ok_termination = termination in (
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.maxTimeLimit,
    )

    if not ((ok_status or aborted_with_solution) and ok_termination):
        return None

    # Verify the model actually has a solution loaded
    try:
        _ = value(pyomo_model.x[next(iter(pyomo_model.Sites))])
    except Exception:
        return None

    solve_time = time.time() - start_time

    # Extract solution
    x_vals = {i: round(value(pyomo_model.x[i])) for i in pyomo_model.Sites}
    y_vals = {j: round(value(pyomo_model.y[j])) for j in pyomo_model.CPs}
    z_vals = {
        (i, j): round(value(pyomo_model.z[i, j]))
        for (i, j) in pyomo_model.Arcs
    }

    # Evaluate solution
    metrics = evaluate_multi_cp_solution(
        x_vals=x_vals,
        y_vals=y_vals,
        z_vals=z_vals,
        energy_per_site=model_inputs['energy_per_site'],
        covariance_matrix=model_inputs['covariance_matrix'],
        fixed_cost=model_inputs['fixed_cost'],
        inter_costs=model_inputs['inter_costs'],
        trans_costs=model_inputs['trans_costs'],
    )

    # For time-limit incumbents, verify LCOE constraint is actually satisfied
    if metrics['achieved_lcoe'] > model_inputs['lcoe_target'] * 1.001:
        return None

    return {
        'success': True,
        'selected_sites': metrics['selected_sites'],
        'active_cps': metrics['active_cps'],
        'assignments': metrics['assignments'],
        'variance': metrics['variance'],
        'achieved_lcoe': metrics['achieved_lcoe'],
        'total_cost': metrics['total_cost'],
        'total_energy': metrics['total_energy'],
        'cost_fixed': metrics['cost_fixed'],
        'cost_inter_array': metrics['cost_inter_array'],
        'cost_transmission': metrics['cost_transmission'],
        'per_cp_details': metrics['per_cp_details'],
        'solve_time': solve_time,
        'solver_used': 'gurobi',
        'message': 'Solved with Gurobi (multi-CP BQP)',
    }
