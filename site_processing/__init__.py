"""
Site Processing Module
======================

Load oceanographic data and filter feasible tidal turbine sites.

Example:
    from tidal_portfolio.site_processing import load_all, process_sites

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
        dist_to_shore=flat['dist_to_shore'],
    )
"""

from .loaders import (
    load_hycom,
    load_gebco,
    load_shoreline,
    load_all,
    flatten_grid_data,
    load_site_results,
)

from .process_sites import (
    process_sites,
    save_sites,
)

from .utide_bridge import (
    prepare_utide_input,
    process_utide_output,
)

from .turbine import (
    load_turbine,
    list_available_turbines,
    TurbineNotFoundError,
)

__all__ = [
    "load_hycom",
    "load_gebco",
    "load_shoreline",
    "load_all",
    "flatten_grid_data",
    "load_site_results",
    "process_sites",
    "save_sites",
    "prepare_utide_input",
    "process_utide_output",
    "load_turbine",
    "list_available_turbines",
    "TurbineNotFoundError",
]
