#!/usr/bin/env python3
"""
Plot HYCOM grid points vs shoreline for failed and zero-site regions.

Shows why certain regions failed: HYCOM points are either on land (failed)
or too far from tidal channels to capture strong currents (zero-site).
"""

import sys
from pathlib import Path
import glob
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR.parent))

import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr

from tidal_portfolio.config import get_region_paths

OUTPUT_DIR = ROOT_DIR / "outputs" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_hycom_grid(hycom_pattern):
    """Load grid from ALL HYCOM tiles (merging _1 and _2) to get lat/lon and NaN mask."""
    files = sorted(glob.glob(hycom_pattern))

    # Group files by tile prefix — take first timestep from each spatial tile
    seen_prefixes = set()
    tile_files = []
    for f in files:
        fname = Path(f).name
        # Extract prefix: everything before the date, e.g. "hycom_Maine_1_"
        parts = fname.rsplit("_", 1)  # split off date
        prefix = parts[0] if len(parts) == 2 else fname
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            tile_files.append(f)

    all_lats = []
    all_lons = []
    all_has_data = []

    for f in tile_files:
        ds = xr.open_dataset(f)

        lats = ds["lat"].values
        lons = ds["lon"].values

        depths = ds["depth"].values
        depth_idx = int(np.argmin(np.abs(depths - 20.0)))
        water_u = ds["water_u"].isel(depth=depth_idx).values

        if water_u.ndim == 3:
            water_u = water_u[0]

        ds.close()

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        has_data = ~np.isnan(water_u)

        all_lats.append(lat_grid.ravel())
        all_lons.append(lon_grid.ravel())
        all_has_data.append(has_data.ravel())

    all_lats = np.concatenate(all_lats)
    all_lons = np.concatenate(all_lons)
    all_has_data = np.concatenate(all_has_data)

    return {
        "lats": all_lats,
        "lons": all_lons,
        "has_data": all_has_data,
        "n_total": len(all_lats),
        "n_valid": int(np.sum(all_has_data)),
    }


def plot_region(region_name, ax):
    """Plot one region: shoreline + HYCOM points colored by data/no-data."""
    paths = get_region_paths(region_name)

    # Load HYCOM grid
    grid = load_hycom_grid(paths["hycom_pattern"])

    # Load shoreline
    shoreline = gpd.read_file(paths["shoreline_path"])

    # Plot shoreline
    shoreline.plot(ax=ax, color="tan", edgecolor="black", linewidth=0.8, alpha=0.6, zorder=1)

    # Plot points WITHOUT data (NaN = on land) in red
    no_data = ~grid["has_data"]
    if np.sum(no_data) > 0:
        ax.scatter(
            grid["lons"][no_data], grid["lats"][no_data],
            c="red", s=8, alpha=0.5, zorder=2, label=f"No data ({np.sum(no_data)})"
        )

    # Plot points WITH data in blue
    if np.sum(grid["has_data"]) > 0:
        ax.scatter(
            grid["lons"][grid["has_data"]], grid["lats"][grid["has_data"]],
            c="blue", s=8, alpha=0.6, zorder=3, label=f"Has data ({grid['n_valid']})"
        )

    pct_missing = (1 - grid["n_valid"] / grid["n_total"]) * 100
    ax.set_title(f"{region_name}\n{grid['n_valid']}/{grid['n_total']} pts with data ({pct_missing:.0f}% missing)",
                 fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


def main():
    # All problem regions
    failed_regions = ["Delaware", "Maryland", "New_York", "Pennsylvania", "Rhode_Island"]
    zero_regions = ["Connecticut", "Maine", "Massachusetts", "New_Hampshire", "New_Jersey", "Virginia"]
    all_regions = failed_regions + zero_regions

    # Create multi-panel figure
    n_regions = len(all_regions)
    n_cols = 3
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 22))
    axes = axes.ravel()

    for i, region in enumerate(all_regions):
        print(f"Plotting {region}...")
        plot_region(region, axes[i])

    # Remove unused subplot
    axes[11].set_visible(False)

    # Add section labels
    fig.text(0.5, 0.98, "Failed Regions (>70% HYCOM data missing — mostly land/bays)",
             ha="center", fontsize=14, fontweight="bold", color="red")
    fig.text(0.5, 0.52, "Zero-Site Regions (data loaded but currents too weak for RM1)",
             ha="center", fontsize=14, fontweight="bold", color="darkblue")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    save_path = OUTPUT_DIR / "failed_regions_hycom_vs_shoreline.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
