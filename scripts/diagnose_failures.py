#!/usr/bin/env python3
"""
Diagnose why regions failed or produced zero viable sites.
"""

import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR.parent))

from tidal_portfolio.site_processing import load_all, flatten_grid_data
from tidal_portfolio.energy.generation import apply_power_curve
from tidal_portfolio.config import get_region_paths
import glob as globmod


# RM1 turbine params
CUT_IN, RATED, CUT_OUT, RATED_POWER = 0.5, 2.0, 3.0, 1.1


def diagnose_failed_regions():
    """Check regions that failed with too much missing data."""
    failed = ['Delaware', 'Maryland', 'New_York', 'Pennsylvania', 'Rhode_Island']

    print("=" * 80)
    print("FAILED REGIONS — Too much missing HYCOM data")
    print("=" * 80)

    for region in failed:
        paths = get_region_paths(region)
        import xarray as xr

        hycom_files = sorted(globmod.glob(paths['hycom_pattern']))
        ds = xr.open_dataset(hycom_files[0])

        # Get velocity variables
        u_var = 'water_u' if 'water_u' in ds else list(ds.data_vars)[0]

        # Count total grid points vs NaN
        data = ds[u_var].values
        total = data.size
        missing = np.sum(np.isnan(data))
        pct = missing / total * 100

        lat_var = 'lat' if 'lat' in ds.dims else 'latitude'
        lon_var = 'lon' if 'lon' in ds.dims else 'longitude'
        n_lat = len(ds[lat_var])
        n_lon = len(ds[lon_var])

        ds.close()

        print(f"\n{region}:")
        print(f"  Grid: {n_lat} x {n_lon} = {n_lat * n_lon} points")
        print(f"  Missing: {pct:.1f}% (threshold: 70%)")
        print(f"  Reason: Most grid points are over land (bays, estuaries, rivers)")

    print()


def diagnose_zero_regions():
    """Deep dive into regions that loaded but had 0 viable sites."""
    zero_regions = ['Connecticut', 'Maine', 'Massachusetts',
                    'New_Hampshire', 'New_Jersey', 'Virginia']

    print("=" * 80)
    print("ZERO-SITE REGIONS — Data loaded but no sites passed filters")
    print("=" * 80)

    for region in zero_regions:
        paths = get_region_paths(region)

        raw = load_all(
            hycom_pattern=paths['hycom_pattern'],
            target_depth_m=20.0,
            max_missing_fraction=0.7,
            gebco_path=paths['gebco_path'],
            shoreline_path=paths['shoreline_path'],
            verbose=False,
        )
        flat = flatten_grid_data(raw)

        speeds = flat['current_speeds']
        depths = flat['depths']
        dist = flat['dist_to_shore']
        mean_speeds = np.nanmean(speeds, axis=1)

        # Compute CFs
        cfs = np.array([
            np.mean(apply_power_curve(speeds[i], CUT_IN, RATED, CUT_OUT, RATED_POWER)) / RATED_POWER
            for i in range(len(mean_speeds))
        ])

        # Filter masks
        valid_depths = ~np.isnan(depths)
        valid_dist = ~np.isnan(dist)
        mask_depth = valid_depths & (depths >= 50) & (depths <= 2000)
        mask_shore = valid_dist & (dist >= 1.0) & (dist <= 200)
        mask_cf = cfs >= 0.05

        print(f"\n{'=' * 60}")
        print(f"{region} — {flat['n_sites']} sites after flattening")
        print(f"{'=' * 60}")

        print(f"\n  Current Speeds:")
        print(f"    Mean of site means:    {np.nanmean(mean_speeds):.4f} m/s")
        print(f"    Max of site means:     {np.nanmax(mean_speeds):.4f} m/s")
        print(f"    Sites mean > 0.5 m/s:  {np.sum(mean_speeds > 0.5)}")
        print(f"    Sites mean > 0.3 m/s:  {np.sum(mean_speeds > 0.3)}")
        print(f"    Sites mean > 0.1 m/s:  {np.sum(mean_speeds > 0.1)}")

        print(f"\n  Depths:")
        vd = depths[valid_depths]
        if len(vd) > 0:
            print(f"    Valid: {len(vd)}/{len(depths)}")
            print(f"    Range: {np.min(vd):.1f} to {np.max(vd):.1f} m")
            print(f"    < 20m:       {np.sum(vd < 20)}")
            print(f"    20-50m:      {np.sum((vd >= 20) & (vd < 50))}")
            print(f"    50-2000m:    {np.sum((vd >= 50) & (vd <= 2000))}")
        else:
            print(f"    No valid depth data!")

        print(f"\n  Capacity Factors (RM1: rated=2.0 m/s):")
        print(f"    Mean CF:        {np.nanmean(cfs):.4f}")
        print(f"    Max CF:         {np.nanmax(cfs):.4f}")
        print(f"    CF > 0.05:      {np.sum(cfs > 0.05)}")
        print(f"    CF > 0.01:      {np.sum(cfs > 0.01)}")

        print(f"\n  Filter Breakdown:")
        print(f"    Pass depth (50-2000m):   {np.sum(mask_depth):4d} / {len(depths)}")
        print(f"    Pass shore (1-200km):    {np.sum(mask_shore):4d} / {len(dist)}")
        print(f"    Pass CF (>=0.05):        {np.sum(mask_cf):4d} / {len(cfs)}")
        print(f"    Pass depth+shore:        {np.sum(mask_depth & mask_shore):4d}")
        print(f"    Pass ALL:                {np.sum(mask_depth & mask_shore & mask_cf):4d}")

        # What's the binding constraint?
        if np.sum(mask_depth & mask_shore) > 0:
            both_cfs = cfs[mask_depth & mask_shore]
            print(f"\n  Sites passing depth+shore ({np.sum(mask_depth & mask_shore)}):")
            print(f"    Their CF range: {np.min(both_cfs):.4f} to {np.max(both_cfs):.4f}")
            print(f"    Their mean speed range: {np.min(mean_speeds[mask_depth & mask_shore]):.4f} to {np.max(mean_speeds[mask_depth & mask_shore]):.4f} m/s")
            print(f"    → CF is the binding constraint" if np.max(both_cfs) < 0.05 else f"    → Some would pass CF too!")
        elif np.sum(mask_depth) == 0:
            print(f"\n  → DEPTH is the binding constraint (no sites in 50-2000m)")
        elif np.sum(mask_shore) == 0:
            print(f"\n  → SHORE DISTANCE is the binding constraint")
        else:
            print(f"\n  → Multiple filters are binding")


if __name__ == "__main__":
    diagnose_failed_regions()
    print("\n\n")
    diagnose_zero_regions()
