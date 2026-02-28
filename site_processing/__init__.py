"""
Site Processing Module
======================

Load oceanographic data and filter feasible tidal turbine sites.

Pipeline:
    1. load_all()          → raw 2D grid data (HYCOM + GEBCO + shoreline)
    2. flatten_grid_data() → 1D site arrays
    3. process_sites()     → filtered feasible sites with power/CF

Optional tidal decomposition (requires MATLAB UTide):
    4. prepare_utide_input()  → .mat files for MATLAB
    5. [MATLAB] run_utide_analysis.m
    6. process_utide_output() → tidal power/CF added to processed data

Example:
    from tidal_portfolio.site_processing import load_all, flatten_grid_data, process_sites

    raw = load_all(hycom_pattern="hycom/*.nc", target_depth_m=20.0,
                   max_missing_fraction=0.7, gebco_path="gebco.nc",
                   shoreline_path="coast.shp")
    flat = flatten_grid_data(raw)
    processed = process_sites(
        latitudes=flat['latitudes'], longitudes=flat['longitudes'],
        current_speeds=flat['current_speeds'], depths=flat['depths'],
        dist_to_shore=flat['dist_to_shore_km'],
        cut_in=0.5, rated=2.0, cut_out=3.0, rated_power=1.1,
        min_depth=20.0, max_depth=100.0,
        min_dist_shore=5.0, max_dist_shore=200.0,
        min_mean_speed=0.5, min_capacity_factor=0.05,
    )
"""

from .loaders import (
    load_all,
    flatten_grid_data,
    load_site_results,
)

from .process_sites import (
    process_sites,
)

from .utide_bridge import (
    prepare_utide_input,
    process_utide_output,
)

from .turbine import (
    load_turbine,
    TurbineNotFoundError,
)

__all__ = [
    "load_all",
    "flatten_grid_data",
    "load_site_results",
    "process_sites",
    "prepare_utide_input",
    "process_utide_output",
    "load_turbine",
    "TurbineNotFoundError",
]
