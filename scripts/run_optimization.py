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
    run_portfolio_optimization, get_best_result, load_turbine, load_site_results,
    save_optimization_results,
)
from tidal_portfolio.config import (
    TURBINES_PER_ARRAY, FCR, WAKE_LOSS_FACTOR,
    ARRAY_ROWS, ARRAY_COLS, ROW_SPACING_M, COL_SPACING_M,
    POWER_FACTOR, INTRA_ARRAY_VOLTAGE_V, INTER_ARRAY_VOLTAGE_V,
    OPEX_RATE,
    get_region_paths,
)
from tidal_portfolio.visualization import plot_all

# =============================================================================
# CONFIGURATION - Modify these parameters as needed
# =============================================================================

# Region name — must match a folder under data/regions/
REGION = "North_Carolina"

# Resolve all data paths for this region
_region = get_region_paths(REGION)

# Turbine model — must match a DEVICE name in data/turbine_specifications.csv
# Available: RM1, RM1-GS, MCT SeaGen, Sabella D10/D15, Atlantis AR1000/AR2000, etc.
TURBINE_NAME = "RM1"

# Input: pre-computed pipeline results from run_energy_pipeline.py
INPUT_NPZ = Path(_region["pipeline_results_dir"]) / f"{TURBINE_NAME}_energy_pipeline_results.npz"

# Number of arrays to deploy
NUM_ARRAYS = 3

# LCOE targets to test ($/MWh)
# Note: Tidal energy LCOE is typically $1000-2000+/MWh at current tech levels
LCOE_TARGETS = [650, 700, 800, 900, 1000]

# Maximum distance from collection point (km)
CLUSTER_RADIUS_KM = 20.0

# Current mode: "total" (all currents) or "tidal" (tidal-only via UTide)
CURRENT_MODE = "tidal"

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
            INPUT_NPZ, require_tidal=(CURRENT_MODE == "tidal"),
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
    print(f"      Turbine (from .npz): {npz_config['turbine_name']}, "
          f"{npz_config['rated_power_mw']} MW rated")

    # -------------------------------------------------------------------------
    # Step 2: Display Configuration
    # -------------------------------------------------------------------------
    print("\n[2/5] Configuration...")
    print(f"      Turbine:           {turbine['name']}")
    print(f"      Rated power:       {turbine['rated_power_mw']} MW")
    print(f"      Turbines/array:    {TURBINES_PER_ARRAY}")
    print(f"      Array capacity:    {TURBINES_PER_ARRAY * turbine['rated_power_mw']:.1f} MW")
    print(f"      Wake loss factor:  {WAKE_LOSS_FACTOR:.0%}")
    print(f"      FCR:               {FCR:.1%}")

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
        capacity_factor=0.35,  # Mean CF estimate for transmission sizing
        opex_rate=OPEX_RATE,
        current_mode=CURRENT_MODE,
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
        if result['feasible']:
            print(f"\n  LCOE Target: ${result['lcoe_target']:.0f}/MWh  [FEASIBLE]")
            print(f"    Achieved LCOE:      ${result['achieved_lcoe']:.0f}/MWh")
            print(f"    Portfolio Variance: {result['variance']:.2f} MW^2")
            print(f"    Net Energy:         {result['energy_net']:,.0f} MWh/year")
            print(f"    Total Cost:         ${result['cost_total']:,.0f}/year")
            print(f"    Transmission Mode:  {result['transmission_mode']}")
            print(f"    Collection Point:   Site {result['collection_point']}")
            print(f"    Selected Sites:     {list(result['selected_sites'])}")

            # Cost breakdown
            print(f"    Cost Breakdown:")
            print(f"      - Fixed:          ${result['cost_fixed']:,.0f}/year")
            print(f"      - Inter-array:    ${result['cost_inter_array']:,.0f}/year")
            print(f"      - Transmission:   ${result['cost_transmission']:,.0f}/year")
        else:
            print(f"\n  LCOE Target: ${result['lcoe_target']:.0f}/MWh  [INFEASIBLE]")

    # Best result summary
    best = get_best_result(results["results"])
    if best:
        print("\n" + "=" * 70)
        print("OPTIMAL SOLUTION (Lowest Feasible LCOE)")
        print("=" * 70)
        print(f"  LCOE:              ${best['achieved_lcoe']:.0f}/MWh")
        print(f"  Portfolio Variance: {best['variance']:.2f} MW^2")
        print(f"  Net Energy:        {best['energy_net']:,.0f} MWh/year")
        print(f"  Total Cost:        ${best['cost_total']:,.0f}/year")
        print(f"  Selected Sites:    {list(best['selected_sites'])}")

        # Get coordinates of selected sites
        print(f"\n  Site Coordinates:")
        for i, site_idx in enumerate(best['selected_sites']):
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
    print("\nGenerate visualization plots? [Y/n]: ", end="")
    response = input().strip().lower()

    if response != "n":
        # Optional: save plots to a directory
        save_dir = Path(_region["plots_dir"]) / "optimization"
        plot_all(site_data, results, save_dir=str(save_dir),
                shoreline_path=_region.get("shoreline_path"))
