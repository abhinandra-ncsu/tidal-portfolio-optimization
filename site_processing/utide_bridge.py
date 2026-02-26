"""
UTide Bridge Module
===================

Functions for bridging between the Python-based tidal portfolio pipeline
and MATLAB UTide harmonic analysis.

Three-step workflow:
    1. prepare_utide_input() — Extract u/v velocities for FILTERED sites → .mat files
    2. [MATLAB] run_utide_analysis.m — Harmonic decomposition → reconstructed velocities
    3. process_utide_output() — Load reconstructed velocities → tidal power/CF

Key design: UTide runs only on sites that passed process_sites() filtering,
not on every grid point. This avoids wasting MATLAB compute on sites that
will never be used (e.g., land, shallow, low-CF).

Example:
    from tidal_portfolio.site_processing import (
        load_all, flatten_grid_data, process_sites,
        prepare_utide_input, process_utide_output,
    )

    raw = load_all(hycom_pattern="hycom/*.nc", ...)
    flat = flatten_grid_data(raw)
    processed = process_sites(flat['latitudes'], flat['current_speeds'], ...)

    # Step 1: Prepare input for MATLAB (only filtered sites)
    prepare_utide_input(raw, processed, output_dir="data/utide_input")

    # Step 2: Run MATLAB (external)
    # >> run_utide_analysis

    # Step 3: Load results back into processed data
    processed = process_utide_output(processed, utide_output_dir="data/utide_output",
                                     cut_in=0.3, rated=1.2, cut_out=3.0, rated_power=1.1)
    # processed now has 'tidal_current_speeds', 'tidal_power_timeseries', 'tidal_capacity_factors'
"""

from pathlib import Path
import numpy as np


# =============================================================================
# DEPENDENCY HELPERS
# =============================================================================

def _require_scipy_io():
    """Get scipy.io module, raising helpful error if not available."""
    try:
        import scipy.io as sio
        return sio
    except ImportError:
        raise ImportError(
            "scipy is required for reading/writing .mat files.\n\n"
            "Install with:\n"
            "    pip install scipy\n\n"
            "Or with conda:\n"
            "    conda install -c conda-forge scipy"
        )


# =============================================================================
# COORDINATE KEY
# =============================================================================

def format_coord_key(lat, lon):
    """
    Create a coordinate-based key for .mat file naming.

    Uses full float precision so keys match exactly when coordinates
    come from the same source (e.g., HYCOM grid values).

    Args:
        lat: Latitude (float)
        lon: Longitude (float)

    Returns:
        String key like "35.04_-75.44"
    """
    return f"{lat}_{lon}"


# =============================================================================
# GRID COORDINATE LOOKUP
# =============================================================================

def _find_grid_indices(site_lat, site_lon, grid_lats, grid_lons):
    """
    Find the nearest grid cell indices for a site coordinate.

    Args:
        site_lat: Site latitude
        site_lon: Site longitude
        grid_lats: 1D array of grid latitudes from raw_data
        grid_lons: 1D array of grid longitudes from raw_data

    Returns:
        (row, col) indices into the 2D grid
    """
    row = int(np.argmin(np.abs(grid_lats - site_lat)))
    col = int(np.argmin(np.abs(grid_lons - site_lon)))
    return row, col


# =============================================================================
# STEP 1: PREPARE INPUT FOR MATLAB UTIDE
# =============================================================================

def prepare_utide_input(raw_data, processed_data, output_dir, verbose=True):
    """
    Prepare per-site .mat files for MATLAB UTide analysis.

    Only processes sites that passed process_sites() filtering — maps each
    filtered site's lat/lon back to the HYCOM grid to extract u/v components.

    Args:
        raw_data: Dict from load_all() — must contain 'water_u', 'water_v',
                  'timestamps', 'latitudes', 'longitudes'
        processed_data: Dict from process_sites() — provides filtered site
                        coordinates via 'latitudes' and 'longitudes'
        output_dir: Directory to save .mat files (will be created if needed)
        verbose: Print progress messages (default: True)

    Returns:
        dict with:
            - n_prepared: Number of .mat files created
            - n_skipped: Number of sites skipped (all-zero velocity)
            - output_dir: Path to output directory
            - coord_keys: List of coordinate keys for prepared sites
    """
    sio = _require_scipy_io()
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get raw 3D velocity grids (n_time, n_lat, n_lon)
    water_u = raw_data['water_u']
    water_v = raw_data['water_v']
    timestamps = raw_data['timestamps']
    grid_lats = raw_data['latitudes']
    grid_lons = raw_data['longitudes']

    # Convert timestamps to formatted strings for MATLAB datenum parsing
    # Format: "dd-mmm-yyyy HH:MM:SS" (e.g., "01-Jan-2020 00:00:00")
    # Note: load_hycom() already deduplicates and sorts timestamps
    formatted_times = pd.to_datetime(timestamps).strftime("%d-%b-%Y %H:%M:%S").values

    # Get filtered site coordinates
    site_lats = processed_data['latitudes']
    site_lons = processed_data['longitudes']
    n_sites = processed_data['n_sites']

    n_prepared = 0
    n_skipped = 0
    coord_keys = []

    if verbose:
        print(f"Preparing UTide input for {n_sites} filtered sites...")
        print(f"  Output directory: {output_dir}")
        print(f"  Timesteps: {len(formatted_times)}")

    for i in range(n_sites):
        lat = float(site_lats[i])
        lon = float(site_lons[i])

        # Map site coordinate → nearest grid cell
        row, col = _find_grid_indices(lat, lon, grid_lats, grid_lons)

        # Extract u/v timeseries for this grid cell
        u_ts = water_u[:, row, col].copy()
        v_ts = water_v[:, row, col].copy()

        # Skip all-zero sites (land points — load_hycom fills NaN with 0)
        if np.all(u_ts == 0) and np.all(v_ts == 0):
            n_skipped += 1
            continue

        # Prepare MATLAB-compatible data
        coord_key = format_coord_key(lat, lon)

        matlab_data = {
            'formatted_times': formatted_times,
            'uin': u_ts.reshape(-1, 1).astype(np.float64),
            'vin': v_ts.reshape(-1, 1).astype(np.float64),
            'lat': lat,
        }

        output_file = output_dir / f"{coord_key}.mat"
        sio.savemat(str(output_file), matlab_data)

        coord_keys.append(coord_key)
        n_prepared += 1

        # Progress update
        if verbose and ((i + 1) % 50 == 0 or (i + 1) == n_sites):
            print(f"  Processed {i + 1}/{n_sites} sites "
                  f"({n_prepared} saved, {n_skipped} skipped)")

    if verbose:
        print(f"  Done: {n_prepared} files saved, {n_skipped} skipped")

    return {
        'n_prepared': n_prepared,
        'n_skipped': n_skipped,
        'output_dir': str(output_dir),
        'coord_keys': coord_keys,
    }


# =============================================================================
# STEP 3: PROCESS MATLAB UTIDE OUTPUT
# =============================================================================

def process_utide_output(processed_data, utide_output_dir,
                         cut_in, rated, cut_out, rated_power,
                         verbose=True):
    """
    Load MATLAB UTide output and compute tidal metrics for filtered sites.

    For each filtered site, loads the reconstructed u/v from MATLAB,
    computes tidal speed, applies the turbine power curve, and derives
    tidal capacity factor. Adds these as new keys to processed_data.

    Args:
        processed_data: Dict from process_sites() — filtered sites
        utide_output_dir: Directory containing UTide output .mat files
        cut_in: Turbine cut-in speed (m/s)
        rated: Turbine rated speed (m/s)
        cut_out: Turbine cut-out speed (m/s)
        rated_power: Turbine rated power (MW)
        verbose: Print progress messages (default: True)

    Returns:
        processed_data dict with added keys:
            - tidal_current_speeds: (n_sites, n_timesteps) tidal-only speeds m/s
            - tidal_power_timeseries: (n_sites, n_timesteps) tidal power MW
            - tidal_capacity_factors: (n_sites,) tidal CF 0-1
            - tidal_mean_speeds_ms: (n_sites,) tidal mean speed m/s
            Sites without UTide output get NaN values.
    """
    from ..energy.generation import apply_power_curve, calculate_capacity_factor

    sio = _require_scipy_io()

    utide_dir = Path(utide_output_dir)
    if not utide_dir.exists():
        raise FileNotFoundError(
            f"UTide output directory not found: {utide_dir}\n"
            "Run run_utide_analysis.m in MATLAB first."
        )

    site_lats = processed_data['latitudes']
    site_lons = processed_data['longitudes']
    n_sites = processed_data['n_sites']
    n_timesteps = processed_data['power_timeseries'].shape[1]

    # Initialize output arrays with NaN
    tidal_speeds = np.full((n_sites, n_timesteps), np.nan)
    tidal_power = np.full((n_sites, n_timesteps), np.nan)
    tidal_cf = np.full(n_sites, np.nan)
    tidal_mean_speed = np.full(n_sites, np.nan)

    n_loaded = 0
    n_missing = 0
    n_errors = 0

    if verbose:
        print(f"Loading UTide output for {n_sites} filtered sites...")
        print(f"  UTide directory: {utide_dir}")

    for i in range(n_sites):
        lat = float(site_lats[i])
        lon = float(site_lons[i])
        coord_key = format_coord_key(lat, lon)
        mat_file = utide_dir / f"{coord_key}.mat"

        if not mat_file.exists():
            n_missing += 1
            continue

        try:
            mat_data = sio.loadmat(str(mat_file))
            u_recon = mat_data['u_recon'].flatten()
            v_recon = mat_data['v_recon'].flatten()

            # Compute tidal speed magnitude
            speed = np.sqrt(u_recon**2 + v_recon**2)

            # Store tidal speed
            tidal_speeds[i, :] = speed
            tidal_mean_speed[i] = np.mean(speed)

            # Apply turbine power curve to tidal speeds
            power_ts = apply_power_curve(speed, cut_in, rated, cut_out, rated_power)
            tidal_power[i, :] = power_ts
            tidal_cf[i] = calculate_capacity_factor(power_ts, rated_power)

            n_loaded += 1

        except Exception as e:
            if verbose:
                print(f"  Error loading {mat_file.name}: {e}")
            n_errors += 1

        # Progress update
        if verbose and ((i + 1) % 50 == 0 or (i + 1) == n_sites):
            print(f"  Processed {i + 1}/{n_sites} sites "
                  f"({n_loaded} loaded, {n_missing} missing, {n_errors} errors)")

    if verbose:
        print(f"  Done: {n_loaded} loaded, {n_missing} missing, {n_errors} errors")

        # Summary statistics for loaded sites
        valid_cf = tidal_cf[~np.isnan(tidal_cf)]
        if len(valid_cf) > 0:
            print(f"  Tidal mean CF:    {np.mean(valid_cf):.4f} ({np.mean(valid_cf):.2%})")
            print(f"  Tidal mean speed: {np.nanmean(tidal_mean_speed):.4f} m/s")

            # Compare to total
            total_cf = processed_data['capacity_factors']
            total_mean = np.mean(total_cf)
            tidal_mean = np.mean(valid_cf)
            if total_mean > 0:
                print(f"  Total mean CF:    {total_mean:.4f} ({total_mean:.2%})")
                print(f"  CF reduction:     {(1 - tidal_mean / total_mean):.1%}")

    # Add tidal data to processed_data
    processed_data['tidal_current_speeds'] = tidal_speeds
    processed_data['tidal_power_timeseries'] = tidal_power
    processed_data['tidal_capacity_factors'] = tidal_cf
    processed_data['tidal_mean_speeds_ms'] = tidal_mean_speed

    return processed_data
