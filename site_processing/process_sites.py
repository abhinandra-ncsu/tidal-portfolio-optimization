"""
Site Processing Functions
=========================

Functions for processing raw oceanographic data into feasible tidal turbine sites.

Filters sites based on:
    - Water depth (suitable for turbine deployment)
    - Distance from shore (within reasonable range)
    - Minimum current speed / capacity factor (viable for energy extraction)

Example:
    from tidal_portfolio.site_processing import load_all, flatten_grid_data, process_sites

    raw = load_all(
        hycom_pattern="hycom/*.nc",
        target_depth_m=20.0,
        max_missing_fraction=0.7,
        gebco_path="gebco.nc",
        shoreline_path="coast.shp",
    )
    flat = flatten_grid_data(raw)
    sites = process_sites(
        latitudes=flat['latitudes'],
        longitudes=flat['longitudes'],
        current_speeds=flat['current_speeds'],
        depths=flat['depths'],
        dist_to_shore=flat['dist_to_shore_km'],
        cut_in=0.5,
        rated=2.0,
        cut_out=3.0,
        rated_power=1.1,
        min_depth=20.0,
        max_depth=100.0,
        min_dist_shore=5.0,
        max_dist_shore=200.0,
        min_mean_speed=0.5,
        min_capacity_factor=0.05,
    )
"""

import numpy as np

# Import power curve from energy module (canonical location)
from ..energy.generation import apply_power_curve, calculate_capacity_factor


def process_sites(latitudes, longitudes, current_speeds, depths, dist_to_shore,
                  cut_in, rated, cut_out, rated_power,
                  min_depth, max_depth, min_dist_shore,
                  max_dist_shore, min_mean_speed, min_capacity_factor,
                  verbose=True):
    """
    Process raw site data into filtered feasible sites.

    Pipeline:
        1. Apply turbine power curve to get power timeseries
        2. Calculate capacity factors
        3. Filter sites based on criteria
        4. Return processed site data as dict

    Args:
        latitudes: Site latitudes (n_sites,)
        longitudes: Site longitudes (n_sites,)
        current_speeds: Current speed timeseries (n_sites, n_timesteps) m/s
        depths: Water depths in meters (n_sites,)
        dist_to_shore: Distance to shore in km (n_sites,)

        Turbine parameters:
        cut_in: Cut-in speed (m/s)
        rated: Rated speed (m/s)
        cut_out: Cut-out speed (m/s)
        rated_power: Rated power (MW)

        Filter criteria:
        min_depth: Minimum depth (m)
        max_depth: Maximum depth (m)
        min_dist_shore: Minimum shore distance (km)
        max_dist_shore: Maximum shore distance (km)
        min_mean_speed: Minimum mean speed (m/s)
        min_capacity_factor: Minimum capacity factor (0–1)

        verbose: Print progress messages (default: True)

    Returns:
        dict with filtered site data:
            - latitudes: Filtered latitudes (n_filtered,)
            - longitudes: Filtered longitudes (n_filtered,)
            - capacity_factors: Capacity factors (n_filtered,)
            - dist_to_shore_km: Shore distances (n_filtered,)
            - power_timeseries: Power timeseries (n_filtered, n_timesteps)
            - depths_m: Water depths (n_filtered,)
            - mean_speeds_ms: Mean current speeds (n_filtered,)
            - n_sites: Number of filtered sites
            - n_raw_sites: Number of raw sites before filtering
    """
    n_sites_raw = len(latitudes)
    if verbose:
        print(f"Processing {n_sites_raw} raw sites...")

    # Lists to collect filtered sites
    filtered_indices = []
    power_series_list = []
    cf_list = []
    mean_speed_list = []

    for i in range(n_sites_raw):
        # Apply power curve
        power_ts = apply_power_curve(current_speeds[i], cut_in, rated, cut_out, rated_power)

        # Calculate metrics
        cf = calculate_capacity_factor(power_ts, rated_power)
        mean_speed = np.mean(current_speeds[i])

        # Apply filters
        if depths[i] < min_depth or depths[i] > max_depth:
            continue
        if dist_to_shore[i] < min_dist_shore or dist_to_shore[i] > max_dist_shore:
            continue
        if mean_speed < min_mean_speed:
            continue
        if cf < min_capacity_factor:
            continue

        filtered_indices.append(i)
        power_series_list.append(power_ts)
        cf_list.append(cf)
        mean_speed_list.append(mean_speed)

    n_filtered = len(filtered_indices)
    if verbose:
        print(f"  Filtered to {n_filtered} feasible sites")

    if n_filtered == 0:
        return {
            'latitudes': np.array([]),
            'longitudes': np.array([]),
            'capacity_factors': np.array([]),
            'dist_to_shore_km': np.array([]),
            'power_timeseries': np.array([]).reshape(0, current_speeds.shape[1]),
            'depths_m': np.array([]),
            'mean_speeds_ms': np.array([]),
            'n_sites': 0,
            'n_raw_sites': n_sites_raw,
        }

    idx = np.array(filtered_indices)

    return {
        'latitudes': np.asarray(latitudes)[idx],
        'longitudes': np.asarray(longitudes)[idx],
        'capacity_factors': np.array(cf_list),
        'dist_to_shore_km': np.asarray(dist_to_shore)[idx],
        'power_timeseries': np.vstack(power_series_list),
        'depths_m': np.asarray(depths)[idx],
        'mean_speeds_ms': np.array(mean_speed_list),
        'n_sites': n_filtered,
        'n_raw_sites': n_sites_raw,
    }

