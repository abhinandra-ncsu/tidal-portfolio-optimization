#!/usr/bin/env python3
"""
Energy Pipeline Visualization
===============================

Loads pre-computed pipeline results from run_energy_pipeline.py and generates
site characterization plots. This separation allows fast iteration on
visualizations without re-running the expensive data loading pipeline.

Usage:
    python scripts/visualize_energy_pipeline.py

Requires:
    matplotlib, numpy
    (optional: geopandas for shoreline overlay on spatial plot)
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR.parent))

from tidal_portfolio.config import HOURS_PER_YEAR, get_region_paths
from tidal_portfolio.site_processing import load_site_results

# =============================================================================
# CONFIGURATION
# =============================================================================

# Region name — must match a folder under data/regions/
REGION = "North_Carolina"

# Resolve all data paths for this region
_region = get_region_paths(REGION)
SHORELINE_PATH = _region["shoreline_path"]

# Input: pipeline results saved by run_energy_pipeline.py
# Filename includes turbine name, e.g. "RM1-GS_energy_pipeline_results.npz"
INPUT_DIR = Path(_region["pipeline_results_dir"])
TURBINE_NAME = "RM1"
INPUT_FILENAME = f"{TURBINE_NAME}_energy_pipeline_results.npz"

# Output: where to save plot images (under outputs/<region>/)
SAVE_DIR = Path(_region["plots_dir"]) / "energy_pipeline"


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_capacity_factor_histogram(processed, save_path=None):
    """
    Plot capacity factor distribution with mean and median lines.

    Args:
        processed: Dict with site data.
        save_path: Optional path to save the figure.
    """
    import matplotlib.pyplot as plt

    cfs = processed["capacity_factors"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cfs, bins=30, alpha=0.7, color="steelblue", edgecolor="black")

    mean_cf = np.mean(cfs)
    median_cf = np.median(cfs)
    ax.axvline(
        mean_cf, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_cf:.1%}"
    )
    ax.axvline(
        median_cf,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_cf:.1%}",
    )

    ax.set_xlabel("Capacity Factor", fontsize=12)
    ax.set_ylabel("Number of Sites", fontsize=12)
    ax.set_title(
        f"Capacity Factor Distribution ({processed['n_sites']} sites)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()


def plot_spatial_capacity_factors(processed, save_path=None):
    """
    Plot spatial map of capacity factors (lon/lat scatter, colored by CF)
    with the shoreline overlaid for geographic context.

    Args:
        processed: Dict with site data.
        save_path: Optional path to save the figure.
    """
    import matplotlib.pyplot as plt

    lats = processed["latitudes"]
    lons = processed["longitudes"]
    cfs = processed["capacity_factors"]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Overlay shoreline if shapefile is available
    try:
        import geopandas as gpd

        shoreline_path = Path(SHORELINE_PATH)
        if shoreline_path.exists():
            shoreline = gpd.read_file(shoreline_path)
            shoreline.plot(
                ax=ax,
                color="tan",
                edgecolor="black",
                linewidth=1.0,
                alpha=0.6,
                zorder=1,
            )
    except ImportError:
        pass  # geopandas not available, skip shoreline

    scatter = ax.scatter(
        lons,
        lats,
        c=cfs,
        cmap="Blues",
        s=50,
        edgecolors="black",
        linewidth=0.3,
        alpha=0.8,
        zorder=2,
    )

    cbar = plt.colorbar(scatter, ax=ax, label="Capacity Factor")
    cbar.ax.set_ylabel("Capacity Factor", fontsize=10)

    ax.set_xlabel("Longitude (°)", fontsize=12)
    ax.set_ylabel("Latitude (°)", fontsize=12)
    ax.set_title(
        f"Spatial Capacity Factors ({processed['n_sites']} sites)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()


def plot_annual_energy_distribution(processed, config, save_path=None):
    """
    Plot histogram of per-turbine annual energy in MWh.

    Args:
        processed: Dict with site data.
        config: Dict with turbine metadata.
        save_path: Optional path to save the figure.
    """
    import matplotlib.pyplot as plt

    cfs = processed["capacity_factors"]
    rated_mw = config["rated_power_mw"]
    energy_mwh = cfs * rated_mw * HOURS_PER_YEAR
    n_sites = processed["n_sites"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(energy_mwh, bins=30, alpha=0.7, color="#27ae60", edgecolor="black")

    mean_e = np.mean(energy_mwh)
    median_e = np.median(energy_mwh)
    ax.axvline(
        mean_e,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_e:.0f} MWh",
    )
    ax.axvline(
        median_e,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_e:.0f} MWh",
    )

    ax.set_xlabel("Annual Energy per Turbine (MWh/year)", fontsize=12)
    ax.set_ylabel("Number of Sites", fontsize=12)
    ax.set_title(
        f"Per-Turbine Annual Energy Distribution ({n_sites} sites)\n"
        f"{config['turbine_name']} — {config['rated_power_mw']} MW rated",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()


# =============================================================================
# PLOT ORCHESTRATOR
# =============================================================================


def generate_all_plots(processed, config, save_dir=None):
    """
    Generate all 4 site characterization plots.

    Args:
        processed: Dict with site data.
        config: Dict with turbine metadata.
        save_dir: Optional directory to save all figures.
    """
    import os

    print("\n" + "=" * 60)
    print("GENERATING SITE CHARACTERIZATION PLOTS")
    print("=" * 60)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 1. Capacity Factor Histogram
    print("\n[1/3] Capacity Factor Distribution...")
    save_path = f"{save_dir}/capacity_factor_histogram.png" if save_dir else None
    plot_capacity_factor_histogram(processed, save_path)

    # 2. Spatial Capacity Factors
    print("\n[2/3] Spatial Capacity Factors...")
    save_path = f"{save_dir}/spatial_capacity_factors.png" if save_dir else None
    plot_spatial_capacity_factors(processed, save_path)

    # 3. Per-Turbine Annual Energy Distribution
    print("\n[3/3] Per-Turbine Annual Energy Distribution...")
    save_path = f"{save_dir}/annual_energy_distribution.png" if save_dir else None
    plot_annual_energy_distribution(processed, config, save_path)

    print("\n" + "=" * 60)
    print("All plots generated!")
    if save_dir:
        print(f"Figures saved to: {save_dir}/")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("ENERGY PIPELINE VISUALIZATION")
    print("=" * 70)

    # Load pre-computed results
    input_path = INPUT_DIR / INPUT_FILENAME
    print(f"\nLoading results from: {input_path}")
    try:
        processed, config = load_site_results(input_path)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    print(f"  Loaded {processed['n_sites']} sites")
    print(f"  Turbine: {config['turbine_name']}, {config['rated_power_mw']} MW rated")

    # Generate all plots
    save_dir = str(SAVE_DIR)
    generate_all_plots(processed, config, save_dir=save_dir)


if __name__ == "__main__":
    main()
