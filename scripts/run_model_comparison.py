#!/usr/bin/env python3
"""
Model Comparison: Min-Variance vs Min-LCOE vs Max-Generation
=============================================================

Runs all three optimization models on the same site data and prints
a side-by-side comparison of selected sites, LCOE, variance, and energy.

Usage:
    python scripts/run_model_comparison.py
"""

import sys
import warnings
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR.parent))

from tidal_portfolio import (
    run_portfolio_optimization,
    run_lcoe_optimization,
    run_generation_optimization,
    get_best_result,
    load_turbine,
    load_site_results,
)
from tidal_portfolio.config import ProjectConfig, get_region_paths

# =============================================================================
# CONFIGURATION
# =============================================================================

REGION = "North_Carolina"
_region = get_region_paths(REGION)

TURBINE_NAME = "RM1"
INPUT_NPZ = Path(_region["pipeline_results_dir"]) / f"{TURBINE_NAME}_energy_pipeline_results.npz"

NUM_ARRAYS = 3
LCOE_TARGETS = [650, 700, 800, 900, 1000]
CLUSTER_RADIUS_KM = 20.0
CURRENT_MODE = "tidal"
CONFIG = ProjectConfig()


# =============================================================================
# COMPARISON OUTPUT
# =============================================================================

def print_model_comparison(results_variance, results_lcoe, results_generation,
                           site_data, num_arrays):
    """Print side-by-side comparison of the three models."""

    # Extract best variance result
    best_var = get_best_result(results_variance['results'])
    best_lcoe = results_lcoe['results'][0] if results_lcoe['results'][0]['feasible'] else None
    best_gen = results_generation['results'][0] if results_generation['results'][0]['feasible'] else None

    models = [
        ("Min-Variance", best_var),
        ("Min-LCOE", best_lcoe),
        ("Max-Generation", best_gen),
    ]

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n{'Metric':<25}", end="")
    for name, _ in models:
        print(f"{name:>18}", end="")
    print()
    print("-" * 79)

    # Feasibility
    print(f"{'Feasible':<25}", end="")
    for _, r in models:
        val = "Yes" if r and r['feasible'] else "No"
        print(f"{val:>18}", end="")
    print()

    # Metrics (only if feasible)
    metrics = [
        ("LCOE ($/MWh)", 'lcoe', "${:.0f}"),
        ("Variance (MW^2)", 'variance', "{:.2f}"),
        ("Total Energy (MWh/yr)", 'total_energy', "{:,.0f}"),
        ("Total Cost ($/yr)", 'total_cost', "${:,.0f}"),
        ("Shore Dist (km)", 'dist_to_shore_km', "{:.1f}"),
        ("Transmission", 'transmission_mode', "{}"),
    ]

    for label, key, fmt in metrics:
        print(f"{label:<25}", end="")
        for _, r in models:
            if r and r['feasible'] and key in r:
                val = fmt.format(r[key])
                print(f"{val:>18}", end="")
            else:
                print(f"{'--':>18}", end="")
        print()

    # Cost breakdown
    print(f"\n{'Cost Breakdown':<25}")
    cost_keys = [
        ("  Fixed", 'fixed'),
        ("  Inter-array", 'inter_array'),
        ("  Transmission", 'transmission'),
    ]
    for label, ckey in cost_keys:
        print(f"{label:<25}", end="")
        for _, r in models:
            if r and r['feasible'] and 'cost_breakdown' in r:
                val = f"${r['cost_breakdown'][ckey]:,.0f}"
                print(f"{val:>18}", end="")
            else:
                print(f"{'--':>18}", end="")
        print()

    # Selected sites
    print(f"\n{'Selected Sites':<25}")
    for name, r in models:
        if r and r['feasible']:
            print(f"\n  {name}:")
            cp = r['collection_point']
            cp_lat = site_data['latitudes'][cp]
            cp_lon = site_data['longitudes'][cp]
            print(f"    CP: ({cp_lat:.4f}, {cp_lon:.4f})")
            for i, si in enumerate(r['selected_sites']):
                lat = site_data['latitudes'][si]
                lon = site_data['longitudes'][si]
                cf = site_data['capacity_factors'][si]
                print(f"    Array {i+1}: ({lat:.4f}, {lon:.4f}) | CF={cf:.1%}")

    # Site overlap
    feasible_models = [(n, r) for n, r in models if r and r['feasible']]
    if len(feasible_models) >= 2:
        print(f"\n{'Site Overlap':<25}")
        for i in range(len(feasible_models)):
            for j in range(i + 1, len(feasible_models)):
                n1, r1 = feasible_models[i]
                n2, r2 = feasible_models[j]
                s1 = set(r1['selected_sites'].tolist())
                s2 = set(r2['selected_sites'].tolist())
                shared = s1 & s2
                print(f"  {n1} ∩ {n2}: {len(shared)}/{num_arrays} sites shared")

    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("TIDAL TURBINE MODEL COMPARISON")
    print("=" * 80)

    # Load data
    print(f"\nLoading site data from {INPUT_NPZ}...")
    turbine = load_turbine(TURBINE_NAME)

    try:
        site_data, npz_config = load_site_results(
            INPUT_NPZ, require_tidal=(CURRENT_MODE == "tidal"),
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        sys.exit(1)

    if npz_config["turbine_name"] != TURBINE_NAME:
        warnings.warn(
            f"Turbine mismatch: .npz was generated with '{npz_config['turbine_name']}' "
            f"but TURBINE_NAME is '{TURBINE_NAME}'.",
            stacklevel=1,
        )

    print(f"  {site_data['n_sites']} feasible sites")
    print(f"  Turbine: {turbine['name']}, {turbine['rated_power_mw']} MW")
    print(f"  Arrays: {NUM_ARRAYS}, Cluster radius: {CLUSTER_RADIUS_KM} km")
    print(f"  Current mode: {CURRENT_MODE}")

    common_kwargs = dict(
        site_data=site_data,
        num_arrays=NUM_ARRAYS,
        rated_power_mw=turbine["rated_power_mw"],
        cluster_radius_km=CLUSTER_RADIUS_KM,
        current_mode=CURRENT_MODE,
        config=CONFIG,
        verbose=True,
    )

    # --- Model 1: Min-Variance (portfolio) ---
    print("\n" + "-" * 80)
    print("MODEL 1: Min-Variance (Portfolio Optimization)")
    print("-" * 80)
    results_variance = run_portfolio_optimization(
        lcoe_targets=LCOE_TARGETS,
        **common_kwargs,
    )

    # --- Model 2: Min-LCOE ---
    print("\n" + "-" * 80)
    print("MODEL 2: Min-LCOE (Enumeration)")
    print("-" * 80)
    results_lcoe = run_lcoe_optimization(**common_kwargs)

    # --- Model 3: Max-Generation ---
    print("\n" + "-" * 80)
    print("MODEL 3: Max-Generation (Enumeration)")
    print("-" * 80)
    results_generation = run_generation_optimization(**common_kwargs)

    # --- Comparison ---
    print_model_comparison(results_variance, results_lcoe, results_generation,
                           site_data, NUM_ARRAYS)

    return results_variance, results_lcoe, results_generation, site_data


if __name__ == "__main__":
    results_variance, results_lcoe, results_generation, site_data = main()

    # -------------------------------------------------------------------------
    # Generate Comparison Plots
    # -------------------------------------------------------------------------
    print("Generate comparison plots? [Y/n]: ", end="")
    response = input().strip().lower()

    if response != "n":
        from tidal_portfolio.visualization import (
            plot_comparison_site_map,
            plot_comparison_metrics,
            plot_comparison_cost_breakdown,
        )
        from tidal_portfolio.config import get_region_paths as _grp

        save_dir = Path(_grp(REGION)["plots_dir"]) / "model_comparison"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Build model_results dict from the three runs
        best_var = get_best_result(results_variance['results'])
        best_lcoe = results_lcoe['results'][0]
        best_gen = results_generation['results'][0]

        model_results = {
            'Min-Variance': best_var,
            'Min-LCOE': best_lcoe,
            'Max-Generation': best_gen,
        }

        shoreline_path = _grp(REGION).get("shoreline_path")

        print("\n[1/3] Site map comparison...")
        plot_comparison_site_map(
            site_data, model_results,
            shoreline_path=shoreline_path,
            save_path=str(save_dir / "comparison_site_map.png"),
        )

        print("[2/3] Metrics comparison...")
        plot_comparison_metrics(
            model_results,
            save_path=str(save_dir / "comparison_metrics.png"),
        )

        print("[3/3] Cost breakdown comparison...")
        plot_comparison_cost_breakdown(
            model_results,
            save_path=str(save_dir / "comparison_cost_breakdown.png"),
        )

        print(f"\nPlots saved to: {save_dir}/")

