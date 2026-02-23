#!/usr/bin/env python3
"""
East Coast Region Sanity Check
===============================

Runs the data loading + site filtering pipeline for every region under
data/regions/ and prints a summary table showing which regions have
viable tidal energy sites.

Uses total currents only (no UTide) for speed.

Usage:
    python scripts/sanity_check_regions.py
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR.parent))

from tidal_portfolio import load_turbine
from tidal_portfolio.site_processing import load_all, flatten_grid_data, process_sites
from tidal_portfolio.config import get_region_paths, REGIONS_DIR

# =============================================================================
# CONFIGURATION — same filter criteria as run_energy_pipeline.py
# =============================================================================

TURBINE_NAME = "RM1"
TARGET_DEPTH_M = 20.0
MAX_MISSING_FRACTION = 0.7

MIN_DEPTH_M = 50
MAX_DEPTH_M = 2000.0
MIN_DIST_SHORE_KM = 1.0
MAX_DIST_SHORE_KM = 200.0
MIN_MEAN_SPEED_MS = 0.0
MIN_CAPACITY_FACTOR = 0.05


def run_region(region_name, turbine):
    """Run pipeline for one region, return summary dict."""
    paths = get_region_paths(region_name)

    t0 = time.time()

    raw_data = load_all(
        hycom_pattern=paths["hycom_pattern"],
        target_depth_m=TARGET_DEPTH_M,
        max_missing_fraction=MAX_MISSING_FRACTION,
        gebco_path=paths["gebco_path"],
        shoreline_path=paths["shoreline_path"],
        verbose=False,
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
        verbose=False,
    )

    elapsed = time.time() - t0

    result = {
        "region": region_name,
        "n_raw": processed["n_raw_sites"],
        "n_viable": processed["n_sites"],
        "time_s": elapsed,
    }

    if processed["n_sites"] > 0:
        result["mean_cf"] = float(np.mean(processed["capacity_factors"]))
        result["max_cf"] = float(np.max(processed["capacity_factors"]))
        result["mean_depth"] = float(np.mean(processed["depths_m"]))
        result["mean_shore_dist"] = float(np.mean(processed["dist_to_shore_km"]))
        result["mean_speed"] = float(np.mean(processed["mean_speeds_ms"]))
    else:
        result["mean_cf"] = 0.0
        result["max_cf"] = 0.0
        result["mean_depth"] = 0.0
        result["mean_shore_dist"] = 0.0
        result["mean_speed"] = 0.0

    return result


def main():
    print("=" * 90)
    print("EAST COAST REGION SANITY CHECK")
    print("=" * 90)

    turbine = load_turbine(TURBINE_NAME)
    print(f"Turbine: {turbine['name']} ({turbine['rated_power_mw']} MW)")
    print(f"Filters: depth {MIN_DEPTH_M}-{MAX_DEPTH_M}m, "
          f"shore {MIN_DIST_SHORE_KM}-{MAX_DIST_SHORE_KM}km, "
          f"min CF {MIN_CAPACITY_FACTOR}")
    print()

    # Discover all regions
    regions = sorted([
        d.name for d in REGIONS_DIR.iterdir()
        if d.is_dir() and (d / "hycom").exists()
    ])
    print(f"Found {len(regions)} regions: {', '.join(regions)}\n")

    results = []
    for i, region in enumerate(regions):
        print(f"[{i+1:2d}/{len(regions)}] {region}...", end=" ", flush=True)
        try:
            r = run_region(region, turbine)
            results.append(r)
            status = f"{r['n_viable']:4d} viable sites (from {r['n_raw']:5d} raw) in {r['time_s']:.1f}s"
            print(status)
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "region": region, "n_raw": 0, "n_viable": 0,
                "mean_cf": 0, "max_cf": 0, "mean_depth": 0,
                "mean_shore_dist": 0, "mean_speed": 0, "time_s": 0,
            })

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    header = (f"{'Region':<20s} {'Raw':>6s} {'Viable':>7s} {'Mean CF':>8s} "
              f"{'Max CF':>7s} {'Depth':>7s} {'Shore':>7s} {'Speed':>7s}")
    print(header)
    print("-" * 90)

    viable_regions = []
    for r in results:
        viable_tag = ""
        if r["n_viable"] > 0:
            viable_regions.append(r["region"])
        else:
            viable_tag = " *"

        row = (f"{r['region']:<20s} {r['n_raw']:6d} {r['n_viable']:7d} "
               f"{r['mean_cf']:7.3f}  {r['max_cf']:6.3f}  "
               f"{r['mean_depth']:6.1f}m {r['mean_shore_dist']:5.1f}km "
               f"{r['mean_speed']:5.3f}{viable_tag}")
        print(row)

    print("-" * 90)
    print(f"* = no viable sites")
    print(f"\nViable regions ({len(viable_regions)}/{len(regions)}): "
          f"{', '.join(viable_regions)}")

    total_viable = sum(r["n_viable"] for r in results)
    print(f"Total viable sites across all regions: {total_viable}")


if __name__ == "__main__":
    main()
