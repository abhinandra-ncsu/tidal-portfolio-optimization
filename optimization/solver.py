"""
Optimization Solver Functions
=============================

Solves the binary quadratic program (BQP) using Gurobi.

Functions:
    - solve_with_gurobi: Solve a built Pyomo model with Gurobi
    - solve_optimization_model: Build + solve in one step
"""

import time
import numpy as np

from pyomo.environ import (
    SolverFactory,
    SolverStatus,
    TerminationCondition,
)

from .model import (
    build_optimization_model,
    get_model_solution,
    evaluate_solution,
)


def solve_with_gurobi(pyomo_model, model_inputs, timeout=300, verbose=False):
    """
    Solve using Gurobi as an exact binary quadratic program.

    Gurobi handles binary variables natively via branch-and-bound,
    so no relaxation or rounding is needed.

    Args:
        pyomo_model: Built Pyomo ConcreteModel (from build_optimization_model)
        model_inputs: Dict with model input arrays, used for solution evaluation.
            Required keys: n_candidates, num_arrays, energy_per_site,
            covariance_matrix, total_fixed_cost, inter_array_costs,
            transmission_cost
        timeout: Solver time limit in seconds
        verbose: Print solver output

    Returns:
        dict with solver result, or None if failed.
        Keys: success, x_values, selected_indices, variance, lcoe,
              total_cost, total_energy, cost_breakdown, solve_time,
              solver_used, message
    """
    solver = SolverFactory('gurobi')
    if solver is None or not solver.available():
        raise RuntimeError("Gurobi solver not available")

    # Set Gurobi parameters
    solver.options['TimeLimit'] = timeout
    solver.options['MIPGap'] = 0.01       # 1% optimality gap
    solver.options['OutputFlag'] = 1 if verbose else 0

    # Solve the model directly (Binary domain, quadratic objective)
    results = solver.solve(pyomo_model, tee=verbose)

    # Check solver status
    status = results.solver.status
    termination = results.solver.termination_condition

    if not (status == SolverStatus.ok and
            termination in (TerminationCondition.optimal,
                            TerminationCondition.feasible)):
        return None

    # Extract binary solution directly (no rounding needed)
    x_values = get_model_solution(pyomo_model)
    x_binary = np.round(x_values).astype(int)

    # Evaluate solution
    metrics = evaluate_solution(
        x=x_binary,
        n_candidates=model_inputs['n_candidates'],
        num_arrays=model_inputs['num_arrays'],
        energy_per_site=model_inputs['energy_per_site'],
        covariance_matrix=model_inputs['covariance_matrix'],
        total_fixed_cost=model_inputs['total_fixed_cost'],
        inter_array_costs=model_inputs['inter_array_costs'],
        transmission_cost=model_inputs['transmission_cost'],
    )

    selected = np.where(x_binary == 1)[0]

    return {
        'success': True,
        'x_values': x_binary,
        'selected_indices': selected,
        'variance': metrics['variance'],
        'lcoe': metrics['lcoe'],
        'total_cost': metrics['total_cost'],
        'total_energy': metrics['total_energy'],
        'cost_breakdown': {
            'fixed': metrics['cost_fixed'],
            'inter_array': metrics['cost_inter_array'],
            'transmission': metrics['cost_transmission'],
        },
        'solve_time': 0.0,  # Set by caller
        'solver_used': 'gurobi',
        'message': 'Solved with Gurobi (exact BQP)',
    }


def solve_optimization_model(model_inputs, timeout=300, verbose=False):
    """
    Build and solve the optimization model using Gurobi.

    This is the main entry point for solving a single optimization
    problem (one collection point, one LCOE target).

    Args:
        model_inputs: Dict with all model inputs.
            Required keys: n_candidates, num_arrays, energy_per_site,
            covariance_matrix, total_fixed_cost, inter_array_costs,
            transmission_cost, lcoe_target
        timeout: Solver timeout in seconds (default: 300)
        verbose: Print solver output

    Returns:
        dict with solver result if successful, None if infeasible.
        See solve_with_gurobi() for result keys.
    """
    start_time = time.time()

    # Build model
    pyomo_model = build_optimization_model(
        n_candidates=model_inputs['n_candidates'],
        num_arrays=model_inputs['num_arrays'],
        energy_per_site=model_inputs['energy_per_site'],
        covariance_matrix=model_inputs['covariance_matrix'],
        total_fixed_cost=model_inputs['total_fixed_cost'],
        inter_array_costs=model_inputs['inter_array_costs'],
        transmission_cost=model_inputs['transmission_cost'],
        lcoe_target=model_inputs['lcoe_target'],
    )

    result = solve_with_gurobi(pyomo_model, model_inputs,
                                timeout=timeout, verbose=verbose)
    if result is not None:
        result['solve_time'] = time.time() - start_time

    return result
