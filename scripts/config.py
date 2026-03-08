"""
Shared Script Configuration
============================

Single source of truth for parameters that must stay in sync across
run_energy_pipeline.py, visualize_energy_pipeline.py, run_optimization.py,
and run_model_comparison.py.

Change REGION here and all scripts pick it up automatically.
The MATLAB script (run_utide_analysis.m) must be updated separately.
"""

# Region name — must match a folder under data/regions/
REGION = "Florida"  # Options: "North_Carolina", "South_Carolina", "Georgia", "Florida"

# Turbine model — must match a DEVICE name in data/turbine_specifications.csv
TURBINE_NAME = "RM1"

# Current mode: "total" (all currents) or "tidal" (tidal-only via UTide)
CURRENT_MODE = "tidal"

# Optimization parameters
NUM_ARRAYS = 3
LCOE_TARGETS = [100, 150, 200, 250, 300, 350, 400, 450, 500]
CLUSTER_RADIUS_KM = 20.0
