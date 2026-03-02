#!/usr/bin/env python3
"""
Tidal Portfolio Optimization - North Carolina
==============================================

Runs the portfolio optimization to find optimal tidal turbine array placements
that minimize portfolio variance while meeting LCOE constraints.

Loads pre-computed site data from an .npz file produced by run_energy_pipeline.py,
so the expensive HYCOM/GEBCO/shoreline pipeline only runs once.

Usage:
    python scripts/run_optimization.py
"""

import sys
import warnings
from pathlib import Path

# Add parent directory to path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR.parent))

from tidal_portfolio import (
    run_portfolio_optimization,
    get_best_result,
    load_turbine,
    load_site_results,
    save_optimization_results,
)
from tidal_portfolio.config import ProjectConfig, get_region_paths
from tidal_portfolio.visualization import plot_all

from config import (
    REGION, TURBINE_NAME, CURRENT_MODE,
    NUM_ARRAYS, LCOE_TARGETS, CLUSTER_RADIUS_KM,
)

# =============================================================================
# CONFIGURATION - Modify shared params in scripts/config.py
# =============================================================================

# Resolve all data paths for this region
_region = get_region_paths(REGION)

# Input: pre-computed pipeline results from run_energy_pipeline.py
INPUT_NPZ = (
    Path(_region["pipeline_results_dir"])
    / f"{TURBINE_NAME}_energy_pipeline_results.npz"
)

# Project configuration (all engineering defaults; override as needed)
CONFIG = ProjectConfig()

# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    print("=" * 70)
    print("TIDAL TURBINE PORTFOLIO OPTIMIZATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load Pre-computed Site Data
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Loading site data from pipeline results...")
    print(f"      Input file:    {INPUT_NPZ}")
    print(f"      Current mode:  {CURRENT_MODE}")

    turbine = load_turbine(TURBINE_NAME)

    try:
        site_data, npz_config = load_site_results(
            INPUT_NPZ,
            require_tidal=(CURRENT_MODE == "tidal"),
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Warn if .npz was generated with a different turbine
    if npz_config["turbine_name"] != TURBINE_NAME:
        warnings.warn(
            f"Turbine mismatch: .npz was generated with '{npz_config['turbine_name']}' "
            f"but TURBINE_NAME is '{TURBINE_NAME}'. Results may be inconsistent. "
            f"Re-run run_energy_pipeline.py with the correct turbine.",
            stacklevel=1,
        )

    print(f"      Loaded {site_data['n_sites']} feasible sites")
    print(
        f"      Turbine (from .npz): {npz_config['turbine_name']}, "
        f"{npz_config['rated_power_mw']} MW rated"
    )

    # -------------------------------------------------------------------------
    # Step 2: Display Configuration
    # -------------------------------------------------------------------------
    print("\n[2/5] Configuration...")
    print(f"      Turbine:           {turbine['name']}")
    print(f"      Rated power:       {turbine['rated_power_mw']} MW")
    print(f"      Turbines/array:    {CONFIG.turbines_per_array}")
    print(
        f"      Array capacity:    {CONFIG.turbines_per_array * turbine['rated_power_mw']:.1f} MW"
    )
    print(f"      Wake loss factor:  {CONFIG.wake_loss_factor:.0%}")
    print(f"      FCR:               {CONFIG.fcr:.1%}")

    # -------------------------------------------------------------------------
    # Step 3: Run Optimization
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Running optimization...")
    print(f"      Arrays to deploy:  {NUM_ARRAYS}")
    print(f"      LCOE targets:      {LCOE_TARGETS} $/MWh")
    print(f"      Cluster radius:    {CLUSTER_RADIUS_KM} km")
    print(f"      Current mode:      {CURRENT_MODE}")

    results = run_portfolio_optimization(
        site_data=site_data,
        num_arrays=NUM_ARRAYS,
        lcoe_targets=LCOE_TARGETS,
        rated_power_mw=turbine["rated_power_mw"],
        cluster_radius_km=CLUSTER_RADIUS_KM,
        current_mode=CURRENT_MODE,
        config=CONFIG,
        verbose=True,
    )

    # -------------------------------------------------------------------------
    # Step 4: Display Results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[4/5] OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\nProject Configuration:")
    print(f"  - Number of arrays:     {results['num_arrays']}")
    print(f"  - Turbines per array:   {results['turbines_per_array']}")
    print(f"  - Project capacity:     {results['project_capacity_mw']:.1f} MW")
    print(f"  - Wake loss factor:     {results['wake_loss_factor']:.0%}")

    print(f"\nResults by LCOE Target:")
    print("-" * 70)

    for result in results["results"]:
        if result["feasible"]:
            print(f"\n  LCOE Target: ${result['lcoe_target']:.0f}/MWh  [FEASIBLE]")
            print(f"    Achieved LCOE:      ${result['lcoe']:.0f}/MWh")
            print(f"    Portfolio Variance: {result['variance']:.2f} MW^2")
            print(f"    Total Energy:       {result['total_energy']:,.0f} MWh/year")
            print(f"    Total Cost:         ${result['total_cost']:,.0f}/year")
            print(f"    Transmission Mode:  {result['transmission_mode']}")
            cp = result["collection_point"]
            cp_lat = site_data["latitudes"][cp]
            cp_lon = site_data["longitudes"][cp]
            print(f"    Collection Point:   ({cp_lat:.4f}, {cp_lon:.4f})")
            print(f"    Selected Sites:")
            for si in result["selected_sites"]:
                print(
                    f"      ({site_data['latitudes'][si]:.4f}, {site_data['longitudes'][si]:.4f})"
                )

            # Cost breakdown
            print(f"    Cost Breakdown:")
            print(
                f"      - Fixed:          ${result['cost_breakdown']['fixed']:,.0f}/year"
            )
            print(
                f"      - Inter-array:    ${result['cost_breakdown']['inter_array']:,.0f}/year"
            )
            print(
                f"      - Transmission:   ${result['cost_breakdown']['transmission']:,.0f}/year"
            )
        else:
            print(f"\n  LCOE Target: ${result['lcoe_target']:.0f}/MWh  [INFEASIBLE]")

    # Best result summary
    best = get_best_result(results["results"])
    if best:
        print("\n" + "=" * 70)
        print("OPTIMAL SOLUTION (Lowest Feasible LCOE)")
        print("=" * 70)
        print(f"  LCOE:              ${best['lcoe']:.0f}/MWh")
        print(f"  Portfolio Variance: {best['variance']:.2f} MW^2")
        print(f"  Total Energy:      {best['total_energy']:,.0f} MWh/year")
        print(f"  Total Cost:        ${best['total_cost']:,.0f}/year")
        print(f"\n  Selected Sites:")
        for i, site_idx in enumerate(best["selected_sites"]):
            lat = site_data["latitudes"][site_idx]
            lon = site_data["longitudes"][site_idx]
            cf = site_data["capacity_factors"][site_idx]
            dist = site_data["dist_to_shore_km"][site_idx]
            print(
                f"    Array {i + 1}: ({lat:.4f}, {lon:.4f}) | CF={cf:.1%} | Shore={dist:.1f}km"
            )
    else:
        print("\n  No feasible solution found for any LCOE target.")
        print("  Try increasing LCOE targets or cluster radius.")

    # -------------------------------------------------------------------------
    # Step 5: Save Results
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Saving results...")
    save_dir = Path(_region["output_dir"]) / "optimization"
    save_optimization_results(
        results=results,
        site_data=site_data,
        turbine=turbine,
        input_npz=INPUT_NPZ,
        output_dir=save_dir,
        turbine_name=TURBINE_NAME,
        region=REGION,
        model="min_variance",
    )

    print("\n" + "=" * 70)
    print("Optimization complete.")
    print("=" * 70)

    return results, site_data


if __name__ == "__main__":
    results, site_data = main()

    # -------------------------------------------------------------------------
    # Generate Visualization Plots
    # -------------------------------------------------------------------------
    save_dir = Path(_region["plots_dir"]) / "optimization"
    plot_all(
        site_data,
        results,
        save_dir=str(save_dir),
        shoreline_path=_region.get("shoreline_path"),
    )
