#!/usr/bin/env python3
"""
Gulf Stream Corridor — Energy Pipeline + Optimization
=======================================================

Runs the full pipeline for all 4 viable East Coast regions:
  1. Energy pipeline (load → flatten → process → save .npz)
  2. Portfolio optimization (load .npz → optimize → save results)

Uses total currents (no UTide) and consistent parameters across all regions.

Usage:
    python scripts/run_gulf_stream_pipeline.py
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR.parent))

from tidal_portfolio import (
    load_turbine, run_portfolio_optimization, get_best_result,
    save_optimization_results,
)
from tidal_portfolio.site_processing import load_all, flatten_grid_data, process_sites
from tidal_portfolio.config import (
    TURBINES_PER_ARRAY, FCR, WAKE_LOSS_FACTOR,
    ARRAY_ROWS, ARRAY_COLS, ROW_SPACING_M, COL_SPACING_M,
    POWER_FACTOR, INTRA_ARRAY_VOLTAGE_V, INTER_ARRAY_VOLTAGE_V,
    OPEX_RATE, HOURS_PER_YEAR,
    get_region_paths,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

REGIONS = ["Florida", "Georgia", "North_Carolina", "South_Carolina"]
TURBINE_NAME = "RM1"

# Energy pipeline params
TARGET_DEPTH_M = 20.0
MAX_MISSING_FRACTION = 0.7
MIN_DEPTH_M = 50
MAX_DEPTH_M = 2000.0
MIN_DIST_SHORE_KM = 1.0
MAX_DIST_SHORE_KM = 200.0
MIN_MEAN_SPEED_MS = 0.0
MIN_CAPACITY_FACTOR = 0.05

# Optimization params
NUM_ARRAYS = 3
LCOE_TARGETS = [650, 700, 800, 900, 1000]
CLUSTER_RADIUS_KM = 20.0
CURRENT_MODE = "total"


# =============================================================================
# ENERGY PIPELINE
# =============================================================================


def run_energy_pipeline(region, turbine):
    """Run energy pipeline for one region, return processed site data."""
    paths = get_region_paths(region)
    output_dir = Path(paths["pipeline_results_dir"])
    output_path = output_dir / f"{TURBINE_NAME}_energy_pipeline_results.npz"

    # Check if already computed
    if output_path.exists():
        print(f"  Loading existing results from {output_path}")
        from tidal_portfolio import load_site_results
        site_data, _ = load_site_results(output_path, require_tidal=False)
        return site_data

    print(f"  Running full pipeline...")

    raw_data = load_all(
        hycom_pattern=paths["hycom_pattern"],
        target_depth_m=TARGET_DEPTH_M,
        max_missing_fraction=MAX_MISSING_FRACTION,
        gebco_path=paths["gebco_path"],
        shoreline_path=paths["shoreline_path"],
        verbose=True,
    )

    flat = flatten_grid_data(raw_data)

    processed = process_sites(
        latitudes=flat["latitudes"],
        longitudes=flat["longitudes"],
        current_speeds=flat["current_speeds"],
        depths=flat["depths"],
        dist_to_shore=flat["dist_to_shore"],
        cut_in=turbine["cut_in_speed_ms"],
        rated=turbine["rated_speed_ms"],
        cut_out=turbine["cut_out_speed_ms"],
        rated_power=turbine["rated_power_mw"],
        min_depth=MIN_DEPTH_M,
        max_depth=MAX_DEPTH_M,
        min_dist_shore=MIN_DIST_SHORE_KM,
        max_dist_shore=MAX_DIST_SHORE_KM,
        min_mean_speed=MIN_MEAN_SPEED_MS,
        min_capacity_factor=MIN_CAPACITY_FACTOR,
        verbose=True,
    )

    if processed["n_sites"] == 0:
        print(f"  WARNING: No viable sites for {region}")
        return None

    # Save .npz
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "latitudes": processed["latitudes"],
        "longitudes": processed["longitudes"],
        "capacity_factors": processed["capacity_factors"],
        "dist_to_shore_km": processed["dist_to_shore_km"],
        "power_timeseries": processed["power_timeseries"],
        "depths_m": processed["depths_m"],
        "mean_speeds_ms": processed["mean_speeds_ms"],
        "n_sites": np.array(processed["n_sites"]),
        "n_raw_sites": np.array(processed["n_raw_sites"]),
        "turbine_name": np.array(turbine["name"]),
        "rated_power_mw": np.array(turbine["rated_power_mw"]),
        "cut_in_speed_ms": np.array(turbine["cut_in_speed_ms"]),
        "rated_speed_ms": np.array(turbine["rated_speed_ms"]),
        "cut_out_speed_ms": np.array(turbine["cut_out_speed_ms"]),
        "rotor_diameter_m": np.array(turbine["rotor_diameter_m"]),
        "n_lat": np.array(raw_data["n_lat"]),
        "n_lon": np.array(raw_data["n_lon"]),
        "n_timesteps": np.array(raw_data["n_timesteps"]),
        "depth_extracted_m": np.array(raw_data["depth_extracted_m"]),
        "lat_min": np.array(raw_data["latitudes"].min()),
        "lat_max": np.array(raw_data["latitudes"].max()),
        "lon_min": np.array(raw_data["longitudes"].min()),
        "lon_max": np.array(raw_data["longitudes"].max()),
        "min_depth_m": np.array(MIN_DEPTH_M),
        "max_depth_m": np.array(MAX_DEPTH_M),
        "min_dist_shore_km": np.array(MIN_DIST_SHORE_KM),
        "max_dist_shore_km": np.array(MAX_DIST_SHORE_KM),
        "min_mean_speed_ms": np.array(MIN_MEAN_SPEED_MS),
        "min_capacity_factor": np.array(MIN_CAPACITY_FACTOR),
        "has_tidal": np.array(False),
    }
    np.savez_compressed(output_path, **save_dict)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to {output_path} ({size_mb:.1f} MB)")

    # Return as site_data dict matching what load_site_results produces
    site_data = {
        "latitudes": processed["latitudes"],
        "longitudes": processed["longitudes"],
        "capacity_factors": processed["capacity_factors"],
        "dist_to_shore": processed["dist_to_shore_km"],
        "power_timeseries": processed["power_timeseries"],
        "depths_m": processed["depths_m"],
        "mean_speeds_ms": processed["mean_speeds_ms"],
        "n_sites": processed["n_sites"],
    }
    return site_data


# =============================================================================
# OPTIMIZATION
# =============================================================================


def run_optimization(region, site_data, turbine):
    """Run portfolio optimization for one region."""
    paths = get_region_paths(region)

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
        capacity_factor=np.mean(site_data["capacity_factors"]),
        opex_rate=OPEX_RATE,
        current_mode=CURRENT_MODE,
        verbose=True,
    )

    # Save results
    save_dir = Path(paths["output_dir"]) / "optimization"
    npz_path = Path(paths["pipeline_results_dir"]) / f"{TURBINE_NAME}_energy_pipeline_results.npz"
    save_optimization_results(
        results=results,
        site_data=site_data,
        turbine=turbine,
        input_npz=npz_path,
        output_dir=save_dir,
        turbine_name=TURBINE_NAME,
        region=region,
    )

    return results


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 80)
    print("GULF STREAM CORRIDOR — MULTI-REGION PIPELINE + OPTIMIZATION")
    print("=" * 80)

    turbine = load_turbine(TURBINE_NAME)
    print(f"Turbine: {turbine['name']} ({turbine['rated_power_mw']} MW)")
    print(f"Arrays: {NUM_ARRAYS} × {TURBINES_PER_ARRAY} turbines = "
          f"{NUM_ARRAYS * TURBINES_PER_ARRAY * turbine['rated_power_mw']:.1f} MW")
    print(f"LCOE targets: {LCOE_TARGETS} $/MWh")
    print(f"Cluster radius: {CLUSTER_RADIUS_KM} km")
    print(f"Current mode: {CURRENT_MODE}")

    all_results = {}

    for i, region in enumerate(REGIONS):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(REGIONS)}] {region.upper()}")
        print(f"{'='*80}")

        # Energy pipeline
        t0 = time.time()
        print(f"\n--- Energy Pipeline ---")
        site_data = run_energy_pipeline(region, turbine)

        if site_data is None:
            print(f"  Skipping optimization (no viable sites)")
            continue

        print(f"  {site_data['n_sites']} viable sites")
        pipeline_time = time.time() - t0

        # Optimization
        t1 = time.time()
        print(f"\n--- Portfolio Optimization ---")
        try:
            results = run_optimization(region, site_data, turbine)
            opt_time = time.time() - t1

            best = get_best_result(results["results"])
            all_results[region] = {
                "results": results,
                "best": best,
                "site_data": site_data,
                "pipeline_time": pipeline_time,
                "opt_time": opt_time,
            }

            if best:
                print(f"\n  Best result: LCOE=${best['achieved_lcoe']:.0f}/MWh, "
                      f"Variance={best['variance']:.2f} MW²")
            else:
                print(f"\n  No feasible solution found")

        except Exception as e:
            print(f"\n  Optimization failed: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # CROSS-REGION COMPARISON
    # =========================================================================
    print(f"\n\n{'='*80}")
    print("CROSS-REGION COMPARISON")
    print(f"{'='*80}")

    print(f"\n{'Region':<20s} {'Sites':>6s} {'Best LCOE':>10s} {'Variance':>10s} "
          f"{'Energy':>12s} {'Total Cost':>12s} {'Tx Mode':>8s}")
    print("-" * 80)

    for region in REGIONS:
        if region not in all_results:
            print(f"{region:<20s} {'—':>6s} {'N/A':>10s}")
            continue

        r = all_results[region]
        best = r["best"]
        n_sites = r["site_data"]["n_sites"]

        if best:
            print(f"{region:<20s} {n_sites:6d} ${best['achieved_lcoe']:8.0f}  "
                  f"{best['variance']:9.2f}  {best['energy_net']:10,.0f}  "
                  f"${best['cost_total']:10,.0f}  {best['transmission_mode']:>8s}")
        else:
            print(f"{region:<20s} {n_sites:6d} {'INFEASIBLE':>10s}")

    # Feasibility matrix
    print(f"\nFeasibility by LCOE target:")
    header = f"{'Region':<20s}" + "".join(f"  ${t:>5.0f}" for t in LCOE_TARGETS)
    print(header)
    print("-" * (20 + 8 * len(LCOE_TARGETS)))

    for region in REGIONS:
        if region not in all_results:
            continue
        row = f"{region:<20s}"
        for result in all_results[region]["results"]["results"]:
            if result["feasible"]:
                row += f"  {'✓':>5s}"
            else:
                row += f"  {'✗':>5s}"
        print(row)

    print(f"\n{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
