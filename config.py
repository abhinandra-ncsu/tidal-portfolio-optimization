"""
Configuration Constants
========================

Central location for all configuration parameters and constants used
throughout the tidal portfolio optimization system.

Module-level constants plus a ``get_region_paths()`` helper that returns
all data paths for a given region.

For turbine loading, see site_processing.turbine.
"""

from pathlib import Path
import glob as _glob


# =============================================================================
# DATA PATHS
# =============================================================================

_PACKAGE_DIR = Path(__file__).parent
DATA_DIR = _PACKAGE_DIR / "data"
OUTPUTS_DIR = _PACKAGE_DIR / "outputs"

REGIONS_DIR = DATA_DIR / "regions"
DEFAULT_REGION = "North_Carolina"

TURBINE_CSV_PATH = str(DATA_DIR / "turbine_specifications.csv")
AC_CABLE_DATA_PATH = str(DATA_DIR / "cables" / "AC_Cables_Param.xlsx")
DC_CABLE_DATA_PATH = str(DATA_DIR / "cables" / "DC_Cables_Param.xlsx")


# =============================================================================
# REGION PATH HELPERS
# =============================================================================


def _find_gebco(gebco_dir):
    """Find the first .nc file in a GEBCO directory."""
    gebco_dir = Path(gebco_dir)
    matches = sorted(gebco_dir.glob("*.nc"))
    if matches:
        return str(matches[0])
    return str(gebco_dir / "*.nc")  # fallback pattern for error messages


def _find_shapefile(shp_dir):
    """Find the first .shp file in a shapefiles directory."""
    shp_dir = Path(shp_dir)
    matches = sorted(shp_dir.glob("*.shp"))
    if matches:
        return str(matches[0])
    return str(shp_dir / "*.shp")  # fallback pattern for error messages


def get_region_paths(region_name=None):
    """
    Return a dict of all data and output paths for a given region.

    Input data lives under ``data/regions/<region>/``, while generated
    outputs (pipeline results, plots) live under ``outputs/<region>/``.

    Parameters
    ----------
    region_name : str or None
        Name of the region folder under ``data/regions/``.
        Defaults to ``DEFAULT_REGION`` ("North_Carolina").

    Returns
    -------
    dict
        Input paths: region_name, region_dir, hycom_pattern, gebco_path,
        shoreline_path.
        Output paths: output_dir, utide_input_dir, utide_output_dir,
        pipeline_results_dir, plots_dir.
    """
    region = region_name or DEFAULT_REGION
    region_dir = REGIONS_DIR / region
    output_dir = OUTPUTS_DIR / region
    return {
        # Input data paths
        "region_name": region,
        "region_dir": str(region_dir),
        "hycom_pattern": str(region_dir / "hycom" / "*.nc"),
        "gebco_path": _find_gebco(region_dir / "gebco"),
        "shoreline_path": _find_shapefile(region_dir / "shapefiles"),
        # Output paths
        "output_dir": str(output_dir),
        "utide_input_dir": str(output_dir / "utide_input"),
        "utide_output_dir": str(output_dir / "utide_output"),
        "pipeline_results_dir": str(output_dir / "pipeline_results"),
        "plots_dir": str(output_dir / "plots"),
    }


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

HOURS_PER_YEAR = 8766  # Average including leap years


# =============================================================================
# ARRAY LAYOUT
# =============================================================================

ARRAY_ROWS = 5
ARRAY_COLS = 8
TURBINES_PER_ARRAY = ARRAY_ROWS * ARRAY_COLS  # 40
ROW_SPACING_M = 300.0  # 15D spacing perpendicular to flow
COL_SPACING_M = 80.0  # ~4D spacing parallel to flow


# =============================================================================
# FINANCIAL PARAMETERS
# =============================================================================

FCR = 0.113  # Fixed Charge Rate (11.3%)
DISCOUNT_RATE = 0.07  # 7% discount rate
PROJECT_LIFETIME = 20  # years
OPEX_RATE = 0.025  # O&M as fraction of CapEx (2.5%)


# =============================================================================
# ELECTRICAL PARAMETERS
# =============================================================================

POWER_FACTOR = 0.95  # AC power factor
INTRA_ARRAY_VOLTAGE_V = 11000  # 11 kV intra-array cables
INTER_ARRAY_VOLTAGE_V = 66000  # 66 kV inter-array cables


# =============================================================================
# ENERGY PARAMETERS
# =============================================================================

WAKE_LOSS_FACTOR = 0.88  # 12% wake loss (multiply by this)
AVAILABILITY = 0.95  # 95% availability
