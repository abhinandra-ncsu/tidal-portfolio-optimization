#!/usr/bin/env python3
"""
Multi-Collection-Point Portfolio Optimization
==============================================

Runs the multi-CP portfolio optimizer that jointly selects K collection points
and N arrays in a single solve, minimizing portfolio variance subject to LCOE
and balance constraints.

Loads pre-computed site data from an .npz file produced by run_energy_pipeline.py.

Usage:
    python scripts/run_multi_cp_optimization.py
"""

import sys
import warnings
from pathlib import Path

# Add parent directory to path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR.parent))

from tidal_portfolio import (
    run_multi_cp_optimization, get_best_multi_cp_result,
    load_turbine, load_site_results,
)
from tidal_portfolio.config import (
    TURBINES_PER_ARRAY, FCR, WAKE_LOSS_FACTOR,
    ARRAY_ROWS, ARRAY_COLS, ROW_SPACING_M, COL_SPACING_M,
    POWER_FACTOR, INTRA_ARRAY_VOLTAGE_V, INTER_ARRAY_VOLTAGE_V,
    OPEX_RATE,
    get_region_paths,
)

# =============================================================================
# CONFIGURATION - Modify these parameters as needed
# =============================================================================

# Region name — must match a folder under data/regions/
REGION = "North_Carolina"

# Resolve all data paths for this region
_region = get_region_paths(REGION)

# Turbine model
TURBINE_NAME = "RM1"

# Input: pre-computed pipeline results
INPUT_NPZ = Path(_region["pipeline_results_dir"]) / f"{TURBINE_NAME}_energy_pipeline_results.npz"

# Number of arrays to deploy
NUM_ARRAYS = 6

# Number of collection points to open
NUM_CPS = 2

# LCOE targets to test ($/MWh)
LCOE_TARGETS = [600, 650, 700, 750, 800, 900, 1000]

# Maximum distance from CP to served array (km)
CLUSTER_RADIUS_KM = 30.0

# Current mode: "total" (all currents) or "tidal" (tidal-only via UTide)
CURRENT_MODE = "tidal"

# Solver timeout per LCOE target (seconds)
SOLVER_TIMEOUT = 600

# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    print("=" * 70)
    print("MULTI-CP TIDAL PORTFOLIO OPTIMIZATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load Pre-computed Site Data
    # -------------------------------------------------------------------------
    print(f"\n[1/4] Loading site data from pipeline results...")
    print(f"      Input file:    {INPUT_NPZ}")
    print(f"      Current mode:  {CURRENT_MODE}")

    turbine = load_turbine(TURBINE_NAME)

    try:
        site_data, npz_config = load_site_results(
            INPUT_NPZ, require_tidal=(CURRENT_MODE == "tidal"),
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    if npz_config["turbine_name"] != TURBINE_NAME:
        warnings.warn(
            f"Turbine mismatch: .npz was generated with '{npz_config['turbine_name']}' "
            f"but TURBINE_NAME is '{TURBINE_NAME}'.",
            stacklevel=1,
        )

    print(f"      Loaded {site_data['n_sites']} feasible sites")
    print(f"      Turbine: {npz_config['turbine_name']}, "
          f"{npz_config['rated_power_mw']} MW rated")

    # -------------------------------------------------------------------------
    # Step 2: Display Configuration
    # -------------------------------------------------------------------------
    print("\n[2/4] Configuration...")
    print(f"      Turbine:           {turbine['name']}")
    print(f"      Rated power:       {turbine['rated_power_mw']} MW")
    print(f"      Turbines/array:    {TURBINES_PER_ARRAY}")
    array_mw = TURBINES_PER_ARRAY * turbine['rated_power_mw']
    print(f"      Array capacity:    {array_mw:.1f} MW")
    print(f"      Arrays (N):        {NUM_ARRAYS}")
    print(f"      CPs (K):           {NUM_CPS}")
    print(f"      Project capacity:  {NUM_ARRAYS * array_mw:.1f} MW")
    print(f"      Wake loss factor:  {WAKE_LOSS_FACTOR:.0%}")
    print(f"      FCR:               {FCR:.1%}")
    print(f"      Cluster radius:    {CLUSTER_RADIUS_KM} km")

    # -------------------------------------------------------------------------
    # Step 3: Run Multi-CP Optimization
    # -------------------------------------------------------------------------
    print(f"\n[3/4] Running multi-CP optimization...")
    print(f"      LCOE targets:      {LCOE_TARGETS} $/MWh")

    results = run_multi_cp_optimization(
        site_data=site_data,
        num_arrays=NUM_ARRAYS,
        num_cps=NUM_CPS,
        lcoe_targets=LCOE_TARGETS,
        wake_loss_factor=WAKE_LOSS_FACTOR,
        turbines_per_array=TURBINES_PER_ARRAY,
        rated_power_mw=turbine["rated_power_mw"],
        cluster_radius_km=CLUSTER_RADIUS_KM,
        fcr=FCR,
        rows=ARRAY_ROWS,
        cols=ARRAY_COLS,
        power_factor=POWER_FACTOR,
        row_spacing_km=ROW_SPACING_M / 1000.0,
        col_spacing_km=COL_SPACING_M / 1000.0,
        intra_array_voltage_v=INTRA_ARRAY_VOLTAGE_V,
        inter_array_voltage_v=INTER_ARRAY_VOLTAGE_V,
        capacity_factor=0.35,
        opex_rate=OPEX_RATE,
        current_mode=CURRENT_MODE,
        timeout=SOLVER_TIMEOUT,
        verbose=True,
    )

    # -------------------------------------------------------------------------
    # Step 4: Display Results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[4/4] OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\nProject Configuration:")
    print(f"  - Arrays (N):           {results['num_arrays']}")
    print(f"  - CPs (K):             {results['num_cps']}")
    print(f"  - Turbines per array:   {results['turbines_per_array']}")
    print(f"  - Project capacity:     {results['project_capacity_mw']:.1f} MW")
    print(f"  - CP candidates:        {results['n_cp_candidates']}")
    print(f"  - Sparse arcs:          {results['n_arcs']}")

    print(f"\nResults by LCOE Target:")
    print("-" * 70)

    for result in results["results"]:
        if result['feasible']:
            print(f"\n  LCOE Target: ${result['lcoe_target']:.0f}/MWh  [FEASIBLE]")
            print(f"    Achieved LCOE:      ${result['achieved_lcoe']:.0f}/MWh")
            print(f"    Portfolio Variance:  {result['variance']:.2f} MW^2")
            print(f"    Total Energy:        {result['total_energy']:,.0f} MWh/year")
            print(f"    Total Cost:          ${result['cost_total']:,.0f}/year")
            print(f"    Selected Sites:      {result['selected_sites']}")
            print(f"    Active CPs:          {result['active_cps']}")
            print(f"    Solve Time:          {result['solve_time']:.1f}s")

            # Cost breakdown
            print(f"    Cost Breakdown:")
            print(f"      - Fixed:           ${result['cost_fixed']:,.0f}/year")
            print(f"      - Inter-array:     ${result['cost_inter_array']:,.0f}/year")
            print(f"      - Transmission:    ${result['cost_transmission']:,.0f}/year")

            # Per-CP details
            print(f"    Per-CP Details:")
            for cp in result['per_cp_details']:
                print(f"      CP {cp['cp_index']}: "
                      f"{cp['num_arrays_served']} arrays, "
                      f"sites={cp['assigned_sites']}, "
                      f"{cp['transmission_mode']}")
        else:
            print(f"\n  LCOE Target: ${result['lcoe_target']:.0f}/MWh  [INFEASIBLE]")

    # Best result summary
    best = get_best_multi_cp_result(results["results"])
    if best:
        print("\n" + "=" * 70)
        print("OPTIMAL SOLUTION (Lowest Feasible LCOE)")
        print("=" * 70)
        print(f"  LCOE:              ${best['achieved_lcoe']:.0f}/MWh")
        print(f"  Portfolio Variance: {best['variance']:.2f} MW^2")
        print(f"  Total Energy:      {best['total_energy']:,.0f} MWh/year")
        print(f"  Total Cost:        ${best['cost_total']:,.0f}/year")
        print(f"  Selected Sites:    {best['selected_sites']}")
        print(f"  Active CPs:        {best['active_cps']}")

        print(f"\n  Site Coordinates:")
        for i, site_idx in enumerate(best['selected_sites']):
            lat = site_data["latitudes"][site_idx]
            lon = site_data["longitudes"][site_idx]
            cf = site_data["capacity_factors"][site_idx]
            dist = site_data["dist_to_shore"][site_idx]
            # Find which CP this site is assigned to
            cp_label = "?"
            for cp_idx, assigned in best['assignments'].items():
                if site_idx in assigned:
                    cp_label = str(cp_idx)
                    break
            print(
                f"    Array {i + 1}: ({lat:.4f}, {lon:.4f}) | "
                f"CF={cf:.1%} | Shore={dist:.1f}km | CP={cp_label}"
            )

        print(f"\n  Collection Points:")
        for cp_detail in best['per_cp_details']:
            j = cp_detail['cp_index']
            lat = site_data["latitudes"][j]
            lon = site_data["longitudes"][j]
            dist = site_data["dist_to_shore"][j]
            print(
                f"    CP {j}: ({lat:.4f}, {lon:.4f}) | "
                f"Shore={dist:.1f}km | "
                f"{cp_detail['transmission_mode']} | "
                f"{cp_detail['num_arrays_served']} arrays"
            )
    else:
        print("\n  No feasible solution found for any LCOE target.")
        print("  Try increasing LCOE targets or cluster radius.")

    # Save results
    save_dir = Path(_region["output_dir"]) / "optimization"
    save_dir.mkdir(parents=True, exist_ok=True)
    import numpy as _np
    save_path = save_dir / f"multi_cp_N{NUM_ARRAYS}_K{NUM_CPS}_{CURRENT_MODE}.npz"
    save_dict = {
        'num_arrays': NUM_ARRAYS,
        'num_cps': NUM_CPS,
        'cluster_radius_km': CLUSTER_RADIUS_KM,
        'current_mode': CURRENT_MODE,
        'lcoe_targets': _np.array(LCOE_TARGETS),
        'n_cp_candidates': results['n_cp_candidates'],
        'n_arcs': results['n_arcs'],
    }
    # Save feasible results
    for idx, r in enumerate(results['results']):
        prefix = f"r{idx}_"
        save_dict[prefix + 'lcoe_target'] = r['lcoe_target']
        save_dict[prefix + 'feasible'] = r['feasible']
        if r['feasible']:
            save_dict[prefix + 'selected_sites'] = _np.array(r['selected_sites'])
            save_dict[prefix + 'active_cps'] = _np.array(r['active_cps'])
            save_dict[prefix + 'variance'] = r['variance']
            save_dict[prefix + 'achieved_lcoe'] = r['achieved_lcoe']
            save_dict[prefix + 'cost_total'] = r['cost_total']
            save_dict[prefix + 'solve_time'] = r['solve_time']

    _np.savez(save_path, **save_dict)
    print(f"\n  Results saved to: {save_path}")

    print("\n" + "=" * 70)
    print("Multi-CP optimization complete.")
    print("=" * 70)

    return results, site_data


if __name__ == "__main__":
    results, site_data = main()
