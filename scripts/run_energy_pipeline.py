#!/usr/bin/env python3
"""
Tidal Energy Site Characterization
====================================

Given ocean current data, bathymetry, and a turbine specification, this script
characterizes each candidate site: capacity factor, per-turbine power time
series, depth, distance to shore, and current speed statistics.

This is the resource assessment stage — it tells you "how good is each site
for a single turbine?" before any array-level or portfolio decisions.

Results are saved to a .npz file for downstream use by visualization or
optimization scripts (see visualize_energy_pipeline.py).

Usage:
    python scripts/run_energy_pipeline.py

Requires:
    xarray, netcdf4, geopandas, shapely, pyproj
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR.parent))

from tidal_portfolio import load_turbine
from tidal_portfolio.site_processing import load_all, flatten_grid_data, process_sites
from tidal_portfolio.config import (
    HOURS_PER_YEAR,
    get_region_paths,
)

# =============================================================================
# CONFIGURATION - Modify these parameters as needed
# =============================================================================

# Region name — must match a folder under data/regions/
REGION = "North_Carolina"

# Resolve all data paths for this region
_region = get_region_paths(REGION)
HYCOM_PATTERN = _region["hycom_pattern"]
GEBCO_PATH = _region["gebco_path"]
SHORELINE_PATH = _region["shoreline_path"]
UTIDE_INPUT_DIR = _region["utide_input_dir"]
UTIDE_OUTPUT_DIR = _region["utide_output_dir"]

# Turbine name (must match a DEVICE in data/turbine_specifications.csv)
#
# "RM1"    — DOE Reference Model 1, designed for fast tidal channels (rated 2.0 m/s)
# "RM1-GS" — RM1 variant re-parameterized for Gulf Stream ocean currents (rated 1.2 m/s)
#             See writing/design_notes/rm1_gs_derivation.md for full derivation.
#
TURBINE_NAME = "RM1"

# Pipeline parameters
# HYCOM extraction depth: rotor center sits at 1D below surface (43 m for RM1-GS).
# Nearest HYCOM depth level is 45.0 m (levels: ...25, 30, 35, 40, 45...).
TARGET_DEPTH_M = 20.0
MAX_MISSING_FRACTION = 0.7  # 70% - accounts for land areas in coastal grids

# Site filter criteria
# The RM1 design requires rotor center submergence of 1D and seabed clearance of 1.5D
# (see literature/RM-1.png). For the RM1-GS (D=43 m): 43 m submergence + 64.5 m seabed
# clearance = 107.5 m minimum water depth. MAX_DEPTH is extended to 2000 m to include Gulf
# Stream sites on the continental slope, assuming a floating platform deployment concept.
# MIN_DIST_SHORE avoids land-contaminated grid cells.
MIN_DEPTH_M = 50  # 1D submergence + 1.5D seabed clearance for D=43 m rotor
MAX_DEPTH_M = 2000.0  # Continental slope; assumes floating platform
MIN_DIST_SHORE_KM = 1.0  # Avoid land-contaminated coastal cells
MAX_DIST_SHORE_KM = 200.0
MIN_MEAN_SPEED_MS = 0.0  # CF filter handles speed screening
MIN_CAPACITY_FACTOR = 0.05

# Current mode: "total" (all currents) or "tidal" (tidal-only via UTide)
# When "tidal", both total and tidal results are computed and saved side-by-side.
# If UTide output is not yet available, prepares .mat input and exits for MATLAB.
CURRENT_MODE = "tidal"

# Output directory for pipeline results (under outputs/<region>/)
OUTPUT_DIR = Path(_region["pipeline_results_dir"])


# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================


def load_and_process_data(turbine):
    """
    Run the full data loading and processing pipeline.

    Stages:
        1. load_all()         → raw grid data (HYCOM + GEBCO + shoreline)
        2. flatten_grid_data() → 1D site arrays
        3. process_sites()     → filtered feasible sites
        4. (if CURRENT_MODE=="tidal") process_utide_output() → tidal metrics

    Args:
        turbine: dict with turbine specs loaded from CSV.

    Returns:
        Tuple of (raw_data, flat, processed, tidal_processed) dicts.
        tidal_processed is None when CURRENT_MODE == "total".
    """
    # Stage 1: Load raw data from all sources
    print("      Running data loading pipeline...")
    raw_data = load_all(
        hycom_pattern=HYCOM_PATTERN,
        target_depth_m=TARGET_DEPTH_M,
        max_missing_fraction=MAX_MISSING_FRACTION,
        gebco_path=GEBCO_PATH,
        shoreline_path=SHORELINE_PATH,
        verbose=True,
    )

    # Stage 2: Flatten grid to 1D arrays
    print("      Flattening grid data...")
    flat = flatten_grid_data(raw_data)

    # Diagnostic: show raw data distributions before filtering
    print("\n      --- Pre-filter data distributions ---")
    valid_depths = flat["depths"][~np.isnan(flat["depths"])]
    print(
        f"      Depths:     {len(valid_depths)}/{flat['n_sites']} valid, "
        f"range [{np.min(valid_depths):.1f}, {np.max(valid_depths):.1f}] m, "
        f"median {np.median(valid_depths):.1f} m"
    )
    in_range = np.sum((valid_depths >= MIN_DEPTH_M) & (valid_depths <= MAX_DEPTH_M))
    print(f"                  {in_range} sites in [{MIN_DEPTH_M}, {MAX_DEPTH_M}] m")
    valid_dist = flat["dist_to_shore"][~np.isnan(flat["dist_to_shore"])]
    print(
        f"      Shore dist: {len(valid_dist)}/{flat['n_sites']} valid, "
        f"range [{np.min(valid_dist):.1f}, {np.max(valid_dist):.1f}] km, "
        f"median {np.median(valid_dist):.1f} km"
    )
    in_range = np.sum(
        (valid_dist >= MIN_DIST_SHORE_KM) & (valid_dist <= MAX_DIST_SHORE_KM)
    )
    print(
        f"                  {in_range} sites in [{MIN_DIST_SHORE_KM}, {MAX_DIST_SHORE_KM}] km"
    )
    mean_speeds_raw = np.mean(flat["current_speeds"], axis=1)
    valid_speeds = mean_speeds_raw[~np.isnan(mean_speeds_raw)]
    print(
        f"      Mean speed: range [{np.min(valid_speeds):.4f}, {np.max(valid_speeds):.4f}] m/s, "
        f"median {np.median(valid_speeds):.4f} m/s"
    )
    above_thresh = np.sum(valid_speeds >= MIN_MEAN_SPEED_MS)
    print(f"                  {above_thresh} sites >= {MIN_MEAN_SPEED_MS} m/s")
    print()

    # Stage 3: Filter sites using turbine params from CSV config (total currents)
    print("      Processing and filtering sites (total currents)...")
    processed = process_sites(
        latitudes=flat["latitudes"],
        longitudes=flat["longitudes"],
        current_speeds=flat["current_speeds"],
        depths=flat["depths"],
        dist_to_shore=flat["dist_to_shore"],
        # Turbine parameters from CSV (not hardcoded defaults)
        cut_in=turbine["cut_in_speed_ms"],
        rated=turbine["rated_speed_ms"],
        cut_out=turbine["cut_out_speed_ms"],
        rated_power=turbine["rated_power_mw"],
        # Filter criteria from script-level constants
        min_depth=MIN_DEPTH_M,
        max_depth=MAX_DEPTH_M,
        min_dist_shore=MIN_DIST_SHORE_KM,
        max_dist_shore=MAX_DIST_SHORE_KM,
        min_mean_speed=MIN_MEAN_SPEED_MS,
        min_capacity_factor=MIN_CAPACITY_FACTOR,
        verbose=True,
    )

    # Stage 4 (optional): UTide auto-detect — prepare input or load output
    if CURRENT_MODE == "tidal":
        from tidal_portfolio.site_processing import prepare_utide_input, process_utide_output
        from tidal_portfolio.site_processing.utide_bridge import format_coord_key

        # Check if UTide output exists for the filtered sites
        utide_out_dir = Path(UTIDE_OUTPUT_DIR)
        sample_key = format_coord_key(processed["latitudes"][0], processed["longitudes"][0])
        utide_ready = (utide_out_dir / f"{sample_key}.mat").exists()

        if not utide_ready:
            # Prepare .mat input files for MATLAB UTide
            print("\n      UTide output not found — preparing input files...")
            summary = prepare_utide_input(
                raw_data=raw_data,
                processed_data=processed,
                output_dir=UTIDE_INPUT_DIR,
                verbose=True,
            )
            # Print MATLAB instructions and exit
            print("\n" + "!" * 70)
            print("MANUAL STEP REQUIRED: Run MATLAB UTide analysis")
            print("!" * 70)
            print(f"  {summary['n_prepared']} .mat files created in {summary['output_dir']}")
            print(f"  1. Open MATLAB")
            print(f"  2. cd '{ROOT_DIR}'")
            print(f"  3. Run: run('scripts/run_utide_analysis')")
            print(f"  4. Re-run: python scripts/run_energy_pipeline.py")
            print("!" * 70)
            sys.exit(0)  # EXIT — do NOT save incomplete .npz
        else:
            # UTide output exists — load tidal data
            print("\n      Loading UTide tidal data for filtered sites...")
            process_utide_output(
                processed_data=processed,
                utide_output_dir=UTIDE_OUTPUT_DIR,
                cut_in=turbine["cut_in_speed_ms"],
                rated=turbine["rated_speed_ms"],
                cut_out=turbine["cut_out_speed_ms"],
                rated_power=turbine["rated_power_mw"],
                verbose=True,
            )

    return raw_data, flat, processed


# =============================================================================
# SUMMARY PRINTING
# =============================================================================


def print_summary(raw_data, flat, processed, turbine):
    """
    Print detailed numerical summary of each pipeline stage.

    All energy metrics are per single turbine (no array scaling, no wake losses).
    When processed contains tidal data (from process_utide_output), also
    prints a tidal vs total comparison.

    Args:
        raw_data: Dict from load_all()
        flat: Dict from flatten_grid_data()
        processed: Dict from process_sites(), optionally with tidal fields
        turbine: dict with turbine specs
    """
    print("\n" + "=" * 70)
    print("SITE CHARACTERIZATION SUMMARY")
    print("=" * 70)

    # --- Data Loading ---
    print("\n--- Data Loading ---")
    print(f"  Grid dimensions:    {raw_data['n_lat']} lat × {raw_data['n_lon']} lon")
    print(f"  Total grid points:  {flat['n_sites']}")
    print(f"  Timesteps:          {raw_data['n_timesteps']}")
    print(f"  HYCOM depth layer:  {raw_data['depth_extracted_m']} m")
    lat_range = (raw_data["latitudes"].min(), raw_data["latitudes"].max())
    lon_range = (raw_data["longitudes"].min(), raw_data["longitudes"].max())
    print(f"  Latitude range:     {lat_range[0]:.4f}° to {lat_range[1]:.4f}°N")
    print(f"  Longitude range:    {lon_range[0]:.4f}° to {lon_range[1]:.4f}°W")

    # --- Filtering ---
    print("\n--- Site Filtering ---")
    n_raw = processed["n_raw_sites"]
    n_filt = processed["n_sites"]
    retention = n_filt / n_raw * 100 if n_raw > 0 else 0
    print(f"  Sites before filter: {n_raw}")
    print(f"  Sites after filter:  {n_filt}")
    print(f"  Retention rate:      {retention:.1f}%")
    print(f"  Filter criteria:")
    print(f"    Depth:             {MIN_DEPTH_M}–{MAX_DEPTH_M} m")
    print(f"    Shore distance:    {MIN_DIST_SHORE_KM}–{MAX_DIST_SHORE_KM} km")
    print(f"    Min mean speed:    {MIN_MEAN_SPEED_MS} m/s")
    print(f"    Min capacity fac:  {MIN_CAPACITY_FACTOR}")

    if n_filt == 0:
        print("\n  WARNING: No sites passed filtering. Check filter criteria.")
        return

    # --- Site Locations ---
    print("\n--- Site Locations (filtered) ---")
    lats = processed["latitudes"]
    lons = processed["longitudes"]
    print(f"  Latitude range:      {np.min(lats):.4f}° to {np.max(lats):.4f}°N")
    print(f"  Longitude range:     {np.min(lons):.4f}° to {np.max(lons):.4f}°W")
    depths = processed["depths_m"]
    dist = processed["dist_to_shore_km"]
    if len(depths) > 0 and not np.all(np.isnan(depths)):
        valid = depths[~np.isnan(depths)]
        print(f"  Depth range:         {np.min(valid):.1f} to {np.max(valid):.1f} m")
        print(f"  Depth median:        {np.median(valid):.1f} m")
    if len(dist) > 0 and not np.all(np.isnan(dist)):
        valid = dist[~np.isnan(dist)]
        print(f"  Shore dist range:    {np.min(valid):.1f} to {np.max(valid):.1f} km")
        print(f"  Shore dist median:   {np.median(valid):.1f} km")

    # --- Current Speeds ---
    print("\n--- Current Speeds (filtered sites) ---")
    mean_speeds = processed["mean_speeds_ms"]
    print(f"  Mean of site means:  {np.mean(mean_speeds):.4f} m/s")
    print(f"  Min site mean:       {np.min(mean_speeds):.4f} m/s")
    print(f"  Max site mean:       {np.max(mean_speeds):.4f} m/s")
    print(f"  Std Dev:             {np.std(mean_speeds):.4f} m/s")

    # --- Capacity Factors ---
    print("\n--- Capacity Factors ---")
    cfs = processed["capacity_factors"]
    print(f"  Mean:      {np.mean(cfs):.4f}  ({np.mean(cfs):.1%})")
    print(f"  Median:    {np.median(cfs):.4f}  ({np.median(cfs):.1%})")
    print(f"  Std Dev:   {np.std(cfs):.4f}")
    print(f"  Min:       {np.min(cfs):.4f}  ({np.min(cfs):.1%})")
    print(f"  Max:       {np.max(cfs):.4f}  ({np.max(cfs):.1%})")
    print(f"  25th pct:  {np.percentile(cfs, 25):.4f}  ({np.percentile(cfs, 25):.1%})")
    print(f"  75th pct:  {np.percentile(cfs, 75):.4f}  ({np.percentile(cfs, 75):.1%})")
    print(f"  90th pct:  {np.percentile(cfs, 90):.4f}  ({np.percentile(cfs, 90):.1%})")

    # --- Per-Turbine Power Output ---
    print("\n--- Per-Turbine Power Output ---")
    power_ts = processed["power_timeseries"]  # (n_sites, n_timesteps) in MW
    rated_mw = turbine["rated_power_mw"]
    mean_power_per_site = np.mean(power_ts, axis=1)
    print(
        f"  Mean power per site:   {np.mean(mean_power_per_site):.4f} MW  ({np.mean(mean_power_per_site) * 1000:.1f} kW)"
    )
    print(
        f"  Max mean power:        {np.max(mean_power_per_site):.4f} MW  ({np.max(mean_power_per_site) * 1000:.1f} kW)"
    )
    # Fraction of time at zero power (below cut-in or above cut-out)
    frac_zero = np.mean(power_ts == 0)
    # Fraction of time at rated power
    frac_rated = np.mean(np.isclose(power_ts, rated_mw, rtol=1e-6))
    print(f"  Time at zero power:    {frac_zero:.1%}")
    print(f"  Time at rated power:   {frac_rated:.1%}")
    print(f"  Time generating:       {1 - frac_zero:.1%}")

    # --- Per-Turbine Annual Energy ---
    print("\n--- Per-Turbine Annual Energy ---")
    # E = CF × P_rated × 8760 hours (single turbine, no wake loss)
    energy_mwh = cfs * rated_mw * HOURS_PER_YEAR
    print(
        f"  Mean:      {np.mean(energy_mwh):.0f} MWh/year  ({np.mean(energy_mwh) / 1000:.2f} GWh)"
    )
    print(f"  Median:    {np.median(energy_mwh):.0f} MWh/year")
    print(f"  Min:       {np.min(energy_mwh):.0f} MWh/year")
    print(f"  Max:       {np.max(energy_mwh):.0f} MWh/year")
    print(f"  Theoretical max:  {rated_mw * HOURS_PER_YEAR:.0f} MWh/year  (CF=100%)")

    # --- Turbine Specification ---
    print("\n--- Turbine Specification ---")
    print(f"  Name:              {turbine['name']}")
    print(
        f"  Rated power:       {turbine['rated_power_mw']} MW  ({turbine['rated_power_mw'] * 1000:.0f} kW)"
    )
    print(f"  Cut-in speed:      {turbine['cut_in_speed_ms']} m/s")
    print(f"  Rated speed:       {turbine['rated_speed_ms']} m/s")
    print(f"  Cut-out speed:     {turbine['cut_out_speed_ms']} m/s")
    print(f"  Rotor diameter:    {turbine['rotor_diameter_m']:.1f} m")

    # --- Tidal vs Total Comparison (when tidal data is available) ---
    has_tidal = "tidal_capacity_factors" in processed
    if has_tidal:
        t_cfs = processed["tidal_capacity_factors"]
        valid_mask = ~np.isnan(t_cfs)

        if np.any(valid_mask):
            rated_mw = turbine["rated_power_mw"]
            t_cfs_valid = t_cfs[valid_mask]
            t_speeds = processed["tidal_mean_speeds_ms"][valid_mask]
            t_power = processed["tidal_power_timeseries"][valid_mask]

            print("\n--- Tidal vs Total Comparison ---")
            print(f"  Current mode:        {CURRENT_MODE}")
            print(f"  Sites with tidal data: {np.sum(valid_mask)}/{n_filt}")

            print(f"\n  Mean capacity factor:")
            print(f"    Total:  {np.mean(cfs):.4f}  ({np.mean(cfs):.1%})")
            print(
                f"    Tidal:  {np.mean(t_cfs_valid):.4f}  ({np.mean(t_cfs_valid):.1%})"
            )
            if np.mean(cfs) > 0:
                cf_reduction = 1 - np.mean(t_cfs_valid) / np.mean(cfs)
                print(f"    Change: {-cf_reduction:+.1%}")

            print(f"\n  Max capacity factor:")
            print(f"    Total:  {np.max(cfs):.4f}  ({np.max(cfs):.1%})")
            print(f"    Tidal:  {np.max(t_cfs_valid):.4f}  ({np.max(t_cfs_valid):.1%})")

            print(f"\n  Mean current speed:")
            print(f"    Total:  {np.mean(mean_speeds):.4f} m/s")
            print(f"    Tidal:  {np.mean(t_speeds):.4f} m/s")

            print(f"\n  Per-turbine annual energy (mean):")
            total_energy = np.mean(cfs) * rated_mw * HOURS_PER_YEAR
            tidal_energy = np.mean(t_cfs_valid) * rated_mw * HOURS_PER_YEAR
            print(f"    Total:  {total_energy:.0f} MWh/year")
            print(f"    Tidal:  {tidal_energy:.0f} MWh/year")

            t_frac_zero = np.mean(t_power == 0)
            print(f"\n  Time at zero power:")
            print(f"    Total:  {frac_zero:.1%}")
            print(f"    Tidal:  {t_frac_zero:.1%}")

    print("\n" + "=" * 70)


# =============================================================================
# SAVING RESULTS
# =============================================================================


def save_results(raw_data, flat, processed, turbine, output_path):
    """
    Save all pipeline outputs to a single .npz file.

    Stores everything needed for downstream visualization and analysis:
    processed site data, config metadata, raw data metadata, and filter params.
    When processed contains tidal data (from process_utide_output), also
    saves tidal-mode results.

    Args:
        raw_data: Dict from load_all().
        flat: Dict from flatten_grid_data().
        processed: Dict from process_sites(), optionally with tidal fields.
        turbine: dict with turbine specs.
        output_path: Path to save the .npz file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the save dict with all standard arrays
    save_dict = {
        # --- Processed site data (for plots and analysis) ---
        "latitudes": processed["latitudes"],
        "longitudes": processed["longitudes"],
        "capacity_factors": processed["capacity_factors"],
        "dist_to_shore_km": processed["dist_to_shore_km"],
        "power_timeseries": processed["power_timeseries"],
        "depths_m": processed["depths_m"],
        "mean_speeds_ms": processed["mean_speeds_ms"],
        "n_sites": np.array(processed["n_sites"]),
        "n_raw_sites": np.array(processed["n_raw_sites"]),
        # --- Config metadata (for plot titles/labels) ---
        "turbine_name": np.array(turbine["name"]),
        "rated_power_mw": np.array(turbine["rated_power_mw"]),
        "cut_in_speed_ms": np.array(turbine["cut_in_speed_ms"]),
        "rated_speed_ms": np.array(turbine["rated_speed_ms"]),
        "cut_out_speed_ms": np.array(turbine["cut_out_speed_ms"]),
        "rotor_diameter_m": np.array(turbine["rotor_diameter_m"]),
        # --- Raw data metadata (for summary reproduction) ---
        "n_lat": np.array(raw_data["n_lat"]),
        "n_lon": np.array(raw_data["n_lon"]),
        "n_timesteps": np.array(raw_data["n_timesteps"]),
        "depth_extracted_m": np.array(raw_data["depth_extracted_m"]),
        "lat_min": np.array(raw_data["latitudes"].min()),
        "lat_max": np.array(raw_data["latitudes"].max()),
        "lon_min": np.array(raw_data["longitudes"].min()),
        "lon_max": np.array(raw_data["longitudes"].max()),
        # --- Filter params (for documentation) ---
        "min_depth_m": np.array(MIN_DEPTH_M),
        "max_depth_m": np.array(MAX_DEPTH_M),
        "min_dist_shore_km": np.array(MIN_DIST_SHORE_KM),
        "max_dist_shore_km": np.array(MAX_DIST_SHORE_KM),
        "min_mean_speed_ms": np.array(MIN_MEAN_SPEED_MS),
        "min_capacity_factor": np.array(MIN_CAPACITY_FACTOR),
    }

    # Add tidal data if available (from process_utide_output)
    has_tidal = "tidal_capacity_factors" in processed
    save_dict["has_tidal"] = np.array(has_tidal)
    if has_tidal:
        save_dict["tidal_capacity_factors"] = processed["tidal_capacity_factors"]
        save_dict["tidal_power_timeseries"] = processed["tidal_power_timeseries"]
        save_dict["tidal_mean_speeds_ms"] = processed["tidal_mean_speeds_ms"]

    np.savez_compressed(output_path, **save_dict)

    # Report file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  Results saved to: {output_path}")
    print(f"  File size: {size_mb:.1f} MB")
    if has_tidal:
        valid_count = np.sum(~np.isnan(processed["tidal_capacity_factors"]))
        print(f"  Includes tidal data: Yes ({valid_count} sites with UTide results)")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("TIDAL ENERGY SITE CHARACTERIZATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load Turbine Specification
    # -------------------------------------------------------------------------
    print("\n[1/3] Loading turbine specification from CSV...")
    print(f"      Turbine: {TURBINE_NAME}")

    turbine = load_turbine(TURBINE_NAME)

    print(
        f"      Rated power:    {turbine['rated_power_mw']} MW ({turbine['rated_power_mw'] * 1000:.0f} kW)"
    )
    print(f"      Cut-in speed:   {turbine['cut_in_speed_ms']} m/s")
    print(f"      Rated speed:    {turbine['rated_speed_ms']} m/s")
    print(f"      Cut-out speed:  {turbine['cut_out_speed_ms']} m/s")

    # -------------------------------------------------------------------------
    # Step 2: Load and Process Site Data
    # -------------------------------------------------------------------------
    print(f"\n[2/3] Loading and processing site data...")
    print(f"      HYCOM pattern: {HYCOM_PATTERN}")
    print(f"      GEBCO file:    {GEBCO_PATH}")
    print(f"      Shoreline:     {SHORELINE_PATH}")
    print(f"      Max missing:   {MAX_MISSING_FRACTION:.0%}")
    print(f"      Current mode:  {CURRENT_MODE}")

    try:
        raw_data, flat, processed = load_and_process_data(turbine)
    except ImportError as e:
        print(f"\nError: Missing dependencies for loading raw data.")
        print(f"Install with: pip install xarray netcdf4 geopandas shapely pyproj")
        print(f"Details: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\nError: Data file not found.")
        print(f"Details: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: Data validation failed.")
        print(f"Details: {e}")
        sys.exit(1)

    print(
        f"\n      Loaded {processed['n_sites']} feasible sites "
        f"(from {processed['n_raw_sites']} raw)"
    )
    if "tidal_capacity_factors" in processed:
        valid_tidal = np.sum(~np.isnan(processed["tidal_capacity_factors"]))
        print(f"      Tidal data: {valid_tidal} sites with UTide results")

    if processed["n_sites"] == 0:
        print("\n  No sites passed filtering. Check filter criteria or data coverage.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Print Summary
    # -------------------------------------------------------------------------
    print_summary(raw_data, flat, processed, turbine)

    # -------------------------------------------------------------------------
    # Step 3: Save Results
    # -------------------------------------------------------------------------
    safe_name = turbine["name"].replace(" ", "_").replace("/", "-")
    output_path = OUTPUT_DIR / f"{safe_name}_energy_pipeline_results.npz"
    print(f"\n[3/3] Saving pipeline results...")
    save_results(raw_data, flat, processed, turbine, output_path)

    return raw_data, flat, processed, turbine


if __name__ == "__main__":
    main()
