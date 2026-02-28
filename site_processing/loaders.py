"""
Data Loaders
============

Functions for loading raw oceanographic data from HYCOM, GEBCO, and shoreline
shapefiles for tidal turbine site assessment.

Main functions:
    - load_all: Load and combine all data sources (HYCOM + GEBCO + shoreline)
    - flatten_grid_data: Convert 2D grid data to 1D arrays for process_sites
    - load_site_results: Load pre-computed pipeline results from .npz

Example:
    from tidal_portfolio.site_processing import load_all, flatten_grid_data

    raw = load_all(
        hycom_pattern="hycom/*.nc",
        target_depth_m=20.0,
        max_missing_fraction=0.7,
        gebco_path="gebco.nc",
        shoreline_path="coast.shp",
    )
    flat = flatten_grid_data(raw)
"""

from pathlib import Path
import glob
import numpy as np


# =============================================================================
# DEPENDENCY HELPERS
# =============================================================================

def _require_xarray():
    """Get xarray module, raising helpful error if not available."""
    try:
        import xarray as xr
        return xr
    except ImportError:
        raise ImportError(
            "xarray is required for loading NetCDF data.\n\n"
            "Install with:\n"
            "    pip install xarray netcdf4\n\n"
            "Or with conda:\n"
            "    conda install -c conda-forge xarray netcdf4"
        )


def _require_geo():
    """Get geopandas, shapely, and pyproj, raising helpful error if not available."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from pyproj import Transformer
        return gpd, Point, Transformer
    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else "geopandas/shapely/pyproj"
        raise ImportError(
            f"{missing} is required for shoreline distance calculations.\n\n"
            "Install with:\n"
            "    pip install geopandas shapely pyproj\n\n"
            "Or with conda:\n"
            "    conda install -c conda-forge geopandas shapely pyproj"
        )


# =============================================================================
# HYCOM LOADER
# =============================================================================

def load_hycom(file_pattern, target_depth_m, max_missing_fraction):
    """
    Load HYCOM ocean current velocity data from NetCDF files.

    Args:
        file_pattern: Glob pattern for NetCDF files (e.g., "hycom/*.nc")
        target_depth_m: Target depth for velocity extraction (m)
        max_missing_fraction: Maximum allowed fraction of missing data (0–1)

    Returns:
        dict with:
            - latitudes: (n_lat,) grid latitudes
            - longitudes: (n_lon,) grid longitudes
            - water_u: (n_time, n_lat, n_lon) eastward velocity m/s
            - water_v: (n_time, n_lat, n_lon) northward velocity m/s
            - timestamps: (n_time,) datetime64 timestamps
            - current_speeds: (n_time, n_lat, n_lon) current speed m/s
            - depth_extracted_m: Actual depth used
            - n_timesteps, n_lat, n_lon: Grid dimensions

    Raises:
        FileNotFoundError: If no files match the pattern
        ValueError: If data has too many missing values
    """
    xr = _require_xarray()

    # Find files
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    # Load dataset
    if len(files) == 1:
        ds = xr.open_dataset(files[0])
    else:
        ds = xr.open_mfdataset(files, combine="by_coords", parallel=True)

    # Find closest depth level
    depths = ds["depth"].values
    depth_idx = int(np.argmin(np.abs(depths - target_depth_m)))
    actual_depth = float(depths[depth_idx])

    # Extract velocity at depth
    water_u = ds["water_u"].isel(depth=depth_idx).values
    water_v = ds["water_v"].isel(depth=depth_idx).values

    # Get coordinates
    latitudes = ds["lat"].values
    longitudes = ds["lon"].values
    timestamps = ds["time"].values

    # Deduplicate latitude dimension — open_mfdataset with combine="by_coords"
    # can leave duplicate rows when spatial tiles share a boundary latitude.
    _, unique_lat_idx = np.unique(latitudes, return_index=True)
    if len(unique_lat_idx) < len(latitudes):
        latitudes = latitudes[unique_lat_idx]
        water_u = water_u[:, unique_lat_idx, :]
        water_v = water_v[:, unique_lat_idx, :]

    # Deduplicate time dimension — spatial tiles share the same timestamps,
    # so open_mfdataset can produce duplicate time entries.
    _, unique_time_idx = np.unique(timestamps, return_index=True)
    if len(unique_time_idx) < len(timestamps):
        timestamps = timestamps[unique_time_idx]
        water_u = water_u[unique_time_idx, :, :]
        water_v = water_v[unique_time_idx, :, :]

    # Check for excessive missing data
    missing_fraction = max(np.sum(np.isnan(water_u)), np.sum(np.isnan(water_v))) / water_u.size
    if missing_fraction > max_missing_fraction:
        raise ValueError(
            f"Too much missing data: {missing_fraction:.1%} missing "
            f"(threshold: {max_missing_fraction:.1%})"
        )

    # Fill missing values with 0 (no current)
    water_u = np.nan_to_num(water_u, nan=0.0)
    water_v = np.nan_to_num(water_v, nan=0.0)

    ds.close()

    # Calculate current speed
    current_speeds = np.sqrt(water_u**2 + water_v**2)

    return {
        'latitudes': latitudes,
        'longitudes': longitudes,
        'water_u': water_u,
        'water_v': water_v,
        'timestamps': timestamps,
        'current_speeds': current_speeds,
        'depth_extracted_m': actual_depth,
        'n_timesteps': len(timestamps),
        'n_lat': len(latitudes),
        'n_lon': len(longitudes),
    }


# =============================================================================
# GEBCO LOADER
# =============================================================================

def load_gebco(file_path, target_lats, target_lons):
    """
    Load GEBCO bathymetry data and interpolate to target grid.

    GEBCO provides global ocean depth data. Elevation values are negative for
    ocean depth; we convert to depth (positive = underwater).

    Args:
        file_path: Path to GEBCO NetCDF file
        target_lats: 1D array of target latitudes
        target_lons: 1D array of target longitudes

    Returns:
        dict with:
            - depths_m: 2D array of depths at target grid (n_lat, n_lon)
            - shape: Shape of the depths array

    Raises:
        FileNotFoundError: If GEBCO file not found
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"GEBCO file not found: {path}")

    xr = _require_xarray()
    ds = xr.open_dataset(path)

    # Create meshgrid for interpolation
    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
    flat_lats = lat_grid.ravel()
    flat_lons = lon_grid.ravel()

    # Interpolate using nearest neighbor
    interp_elevation = ds["elevation"].interp(
        {"lat": xr.DataArray(flat_lats, dims="points"),
         "lon": xr.DataArray(flat_lons, dims="points")},
        method="nearest",
    ).values

    ds.close()

    # Reshape and convert elevation to depth (GEBCO: negative = underwater)
    depths = -interp_elevation.reshape(len(target_lats), len(target_lons))

    return {
        'depths_m': depths,
        'shape': depths.shape,
    }


# =============================================================================
# SHORELINE LOADER
# =============================================================================

def _get_utm_epsg(lon, lat):
    """Get EPSG code for UTM zone from coordinates."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone  # Northern hemisphere
    else:
        return 32700 + zone  # Southern hemisphere


def load_shoreline(shapefile_path, latitudes, longitudes):
    """
    Load shoreline and calculate distances from each grid point to shore.

    Uses UTM projection for accurate distance calculations, automatically
    detecting the appropriate UTM zone from the data centroid.

    Args:
        shapefile_path: Path to shoreline shapefile
        latitudes: 1D array of target latitudes
        longitudes: 1D array of target longitudes

    Returns:
        dict with:
            - distances_km: 2D array of distances in km (n_lat, n_lon)
            - utm_epsg: EPSG code used for UTM projection
            - shape: Shape of the distances array

    Raises:
        FileNotFoundError: If shapefile not found
    """
    path = Path(shapefile_path)
    if not path.exists():
        raise FileNotFoundError(f"Shapefile not found: {path}")

    gpd, Point, Transformer = _require_geo()

    # Load shoreline
    shoreline = gpd.read_file(path)

    # Get UTM zone from data centroid
    center_lon = (longitudes.min() + longitudes.max()) / 2
    center_lat = (latitudes.min() + latitudes.max()) / 2
    utm_epsg = _get_utm_epsg(center_lon, center_lat)

    # Transform shoreline to UTM
    shoreline_utm = shoreline.to_crs(epsg=utm_epsg)
    combined = shoreline_utm.unary_union

    # Create coordinate transformer
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)

    # Transform all grid points to UTM
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    flat_lons = lon_grid.ravel()
    flat_lats = lat_grid.ravel()
    utm_x, utm_y = transformer.transform(flat_lons, flat_lats)

    # Calculate distance from each point to shoreline
    n_lat, n_lon = len(latitudes), len(longitudes)
    distances = np.zeros((n_lat, n_lon))

    for i in range(len(utm_x)):
        point = Point(utm_x[i], utm_y[i])
        dist_m = combined.distance(point)
        row, col = i // n_lon, i % n_lon
        distances[row, col] = dist_m / 1000.0  # Convert to km

    return {
        'distances_km': distances,
        'utm_epsg': utm_epsg,
        'shape': distances.shape,
    }


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def load_all(hycom_pattern, target_depth_m, max_missing_fraction,
             gebco_path, shoreline_path, verbose=True):
    """
    Load all data sources and combine into a single dict.

    This is the main entry point for loading raw oceanographic data.
    Orchestrates HYCOM, GEBCO, and shoreline loaders.

    Args:
        hycom_pattern: Glob pattern for HYCOM NetCDF files
        target_depth_m: Target depth for velocity extraction (m)
        max_missing_fraction: Max allowed missing data in HYCOM (0–1)
        gebco_path: Path to GEBCO NetCDF file
        shoreline_path: Path to shoreline shapefile
        verbose: Print progress messages (default: True)

    Returns:
        dict with:
            - latitudes: (n_lat,) grid latitudes
            - longitudes: (n_lon,) grid longitudes
            - water_u: (n_time, n_lat, n_lon) eastward velocity m/s
            - water_v: (n_time, n_lat, n_lon) northward velocity m/s
            - timestamps: (n_time,) datetime64 timestamps
            - current_speeds: (n_time, n_lat, n_lon) current speed m/s
            - depths_m: (n_lat, n_lon) water depth in meters
            - dist_to_shore_km: (n_lat, n_lon) shore distance in km
            - n_timesteps, n_lat, n_lon: Grid dimensions
            - depth_extracted_m: Actual HYCOM depth used
    """
    # Step 1: Load HYCOM data
    if verbose:
        print("Loading HYCOM data...")

    hycom = load_hycom(hycom_pattern, target_depth_m, max_missing_fraction)

    if verbose:
        print(f"  Loaded: {hycom['n_timesteps']} timesteps, "
              f"{hycom['n_lat']}x{hycom['n_lon']} grid, "
              f"depth={hycom['depth_extracted_m']}m")

    # Step 2: Load GEBCO depths
    if verbose:
        print("Loading GEBCO bathymetry...")

    gebco = load_gebco(
        gebco_path,
        target_lats=hycom['latitudes'],
        target_lons=hycom['longitudes'],
    )

    if verbose:
        print(f"  Loaded: {gebco['shape']} grid")

    # Step 3: Calculate shore distances
    if verbose:
        print("Calculating shoreline distances...")

    shoreline = load_shoreline(
        shoreline_path,
        latitudes=hycom['latitudes'],
        longitudes=hycom['longitudes'],
    )

    if verbose:
        print(f"  Loaded: {shoreline['shape']} grid, UTM EPSG={shoreline['utm_epsg']}")

    if verbose:
        print(f"Pipeline complete: grid={hycom['n_lat']}x{hycom['n_lon']}, "
              f"timesteps={hycom['n_timesteps']}")

    return {
        'latitudes': hycom['latitudes'],
        'longitudes': hycom['longitudes'],
        'water_u': hycom['water_u'],
        'water_v': hycom['water_v'],
        'timestamps': hycom['timestamps'],
        'current_speeds': hycom['current_speeds'],
        'depths_m': gebco['depths_m'],
        'dist_to_shore_km': shoreline['distances_km'],
        'n_timesteps': hycom['n_timesteps'],
        'n_lat': hycom['n_lat'],
        'n_lon': hycom['n_lon'],
        'depth_extracted_m': hycom['depth_extracted_m'],
    }


def load_site_results(npz_path, require_tidal=False):
    """
    Load pre-computed pipeline results from an .npz file.

    Loads site data saved by run_energy_pipeline.py, returning a tuple of
    (site_data, config) dicts ready for optimization or visualization.

    Args:
        npz_path: Path to the .npz file from run_energy_pipeline.py.
        require_tidal: If True, raise ValueError when tidal data is missing.

    Returns:
        (site_data, config) tuple where:
        - site_data: dict with latitudes, longitudes, capacity_factors,
          dist_to_shore_km, power_timeseries, depths_m,
          n_sites, n_raw_sites. When tidal data is present,
          also includes tidal_capacity_factors and tidal_power_timeseries.
        - config: dict with turbine_name, rated_power_mw, cut_in_speed_ms,
          rated_speed_ms, cut_out_speed_ms, rotor_diameter_m.

    Raises:
        FileNotFoundError: If npz_path does not exist.
        ValueError: If require_tidal=True but no tidal data in file.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Pipeline results not found: {npz_path}\n"
            "Run 'python scripts/run_energy_pipeline.py' first."
        )

    data = np.load(npz_path, allow_pickle=False)

    site_data = {
        "latitudes": data["latitudes"],
        "longitudes": data["longitudes"],
        "capacity_factors": data["capacity_factors"],
        "dist_to_shore_km": data["dist_to_shore_km"],
        "power_timeseries": data["power_timeseries"],
        "depths_m": data["depths_m"],
        "n_sites": int(data["n_sites"]),
        "n_raw_sites": int(data["n_raw_sites"]),
    }

    has_tidal = bool(data["has_tidal"]) if "has_tidal" in data else False

    if require_tidal and not has_tidal:
        raise ValueError(
            "Tidal data not found in pipeline results.\n"
            "Re-run 'python scripts/run_energy_pipeline.py' with "
            "CURRENT_MODE = 'tidal' after completing UTide analysis."
        )

    if has_tidal:
        site_data["tidal_capacity_factors"] = data["tidal_capacity_factors"]
        site_data["tidal_power_timeseries"] = data["tidal_power_timeseries"]

    config = {
        "turbine_name": str(data["turbine_name"]),
        "rated_power_mw": float(data["rated_power_mw"]),
        "cut_in_speed_ms": float(data["cut_in_speed_ms"]),
        "rated_speed_ms": float(data["rated_speed_ms"]),
        "cut_out_speed_ms": float(data["cut_out_speed_ms"]),
        "rotor_diameter_m": float(data["rotor_diameter_m"]),
    }

    return site_data, config


def flatten_grid_data(raw_data):
    """
    Flatten 2D grid data to 1D arrays for site processing.

    Converts grid-structured data (n_lat, n_lon) to flat arrays (n_sites,)
    suitable for the process_sites function.

    Args:
        raw_data: Dict from load_all()

    Returns:
        dict with:
            - latitudes: (n_sites,) flat latitudes
            - longitudes: (n_sites,) flat longitudes
            - current_speeds: (n_sites, n_timesteps) flat current speeds
            - depths: (n_sites,) flat depths in meters
            - dist_to_shore_km: (n_sites,) flat shore distances in km
            - n_sites: Total number of sites
            - n_timesteps: Number of timesteps
    """
    lats = raw_data['latitudes']
    lons = raw_data['longitudes']

    # Create meshgrid and flatten
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    flat_lats = lat_grid.ravel()
    flat_lons = lon_grid.ravel()

    n_sites = len(flat_lats)

    # Flatten current speeds: (n_time, n_lat, n_lon) -> (n_sites, n_time)
    current_speeds = raw_data['current_speeds']
    n_time = current_speeds.shape[0]
    flat_speeds = current_speeds.reshape(n_time, -1).T

    return {
        'latitudes': flat_lats,
        'longitudes': flat_lons,
        'current_speeds': flat_speeds,
        'depths': raw_data['depths_m'].ravel(),
        'dist_to_shore_km': raw_data['dist_to_shore_km'].ravel(),
        'n_sites': n_sites,
        'n_timesteps': n_time,
    }
