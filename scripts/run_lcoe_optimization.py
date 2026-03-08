#!/usr/bin/env python3
"""
Min-LCOE Optimization
======================

Runs the min-LCOE enumeration model and saves results to disk.

Usage:
    python scripts/run_lcoe_optimization.py
"""

import sys
import warnings
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR.parent))

from tidal_portfolio import (
    run_lcoe_optimization,
    load_turbine,
    load_site_results,
    save_optimization_results,
)
from tidal_portfolio.config import ProjectConfig, get_region_paths
from tidal_portfolio.visualization import plot_site_map

from config import (
    REGION, TURBINE_NAME, CURRENT_MODE,
    NUM_ARRAYS, CLUSTER_RADIUS_KM,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

_region = get_region_paths(REGION, CURRENT_MODE)
INPUT_NPZ = (
    Path(_region["pipeline_results_dir"])
    / f"{TURBINE_NAME}_energy_pipeline_results.npz"
)
CONFIG = ProjectConfig()

# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("TIDAL TURBINE MIN-LCOE OPTIMIZATION")
    print("=" * 70)

    # Load site data
    print(f"\n[1/4] Loading site data from pipeline results...")
    print(f"      Input file:    {INPUT_NPZ}")
    print(f"      Current mode:  {CURRENT_MODE}")

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

    print(f"      Loaded {site_data['n_sites']} feasible sites")

    # Configuration
    print(f"\n[2/4] Configuration...")
    print(f"      Turbine:           {turbine['name']}")
    print(f"      Arrays to deploy:  {NUM_ARRAYS}")
    print(f"      Cluster radius:    {CLUSTER_RADIUS_KM} km")
    print(f"      Current mode:      {CURRENT_MODE}")

    # Run optimization
    print(f"\n[3/4] Running min-LCOE enumeration...")
    results = run_lcoe_optimization(
        site_data=site_data,
        num_arrays=NUM_ARRAYS,
        rated_power_mw=turbine["rated_power_mw"],
        cluster_radius_km=CLUSTER_RADIUS_KM,
        current_mode=CURRENT_MODE,
        config=CONFIG,
        verbose=True,
    )

    # Display result
    r = results["results"][0]
    if r["feasible"]:
        print(f"\n  Min-LCOE Result:")
        print(f"    LCOE:            ${r['lcoe']:.0f}/MWh")
        print(f"    Variance:        {r['variance']:.2f} MW^2")
        print(f"    Total Energy:    {r['total_energy']:,.0f} MWh/year")
        print(f"    Total Cost:      ${r['total_cost']:,.0f}/year")
        print(f"    Transmission:    {r['transmission_mode']}")
        cp = r["collection_point"]
        print(f"    Collection Pt:   ({site_data['latitudes'][cp]:.4f}, {site_data['longitudes'][cp]:.4f})")
        print(f"    Selected Sites:")
        for i, si in enumerate(r["selected_sites"]):
            lat = site_data["latitudes"][si]
            lon = site_data["longitudes"][si]
            if CURRENT_MODE == "tidal" and "tidal_capacity_factors" in site_data:
                cf = site_data["tidal_capacity_factors"][si]
            else:
                cf = site_data["capacity_factors"][si]
            print(f"      Array {i+1}: ({lat:.4f}, {lon:.4f}) | CF={cf:.1%}")
    else:
        print("\n  No feasible solution found.")

    # Save
    print(f"\n[4/4] Saving results...")
    save_dir = Path(_region["mode_output_dir"]) / "optimization"
    save_optimization_results(
        results=results,
        site_data=site_data,
        turbine=turbine,
        input_npz=INPUT_NPZ,
        output_dir=save_dir,
        turbine_name=TURBINE_NAME,
        region=REGION,
        model="min_lcoe",
    )

    print("\n" + "=" * 70)
    print("Min-LCOE optimization complete.")
    print("=" * 70)

    return results, site_data


if __name__ == "__main__":
    results, site_data = main()

    save_dir = Path(_region["plots_dir"]) / "optimization"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_site_map(
        site_data,
        results["results"],
        shoreline_path=_region.get("shoreline_path"),
        save_path=str(save_dir / "min_lcoe_site_map.png"),
    )
