"""
Electrical Infrastructure Cost Functions
========================================

Functions for calculating electrical infrastructure costs:
    - Intra-array: Cabling within a turbine array
    - Inter-array: Cabling from array to collection point
    - Transmission: Export cable from collection point to shore
"""

# Shared helpers (single canonical source)
from .helpers import (
    calculate_mva,
    calculate_current,
    calculate_cable_cost_per_km,
)

# Coefficient data
from .coefficients import (
    CABLE_COST_COEFFICIENTS,
    INTRA_ARRAY_TRANSFORMER_COEFFICIENTS,
    INTER_ARRAY_TRANSFORMER_COEFFICIENTS,
    SUBSTATION_ANCILLARY_COST,
)

# Intra-array functions
from .intra_array import (
    calculate_intra_array_cost,
    calculate_intra_array_cost_for_arrays,
    calculate_transformer_cost,
    calculate_string_cable_cost,
)

# Inter-array functions
from .inter_array import (
    calculate_inter_array_cost,
    calculate_single_inter_array_cost,
    calculate_substation_transformer_cost,
)

# Transmission functions (physics-based)
from .transmission import (
    calculate_transmission_cost,
    calculate_hvac_cost,
    calculate_hvdc_cost,
    # Cable data loaders
    load_ac_cable_data,
    load_dc_cable_data,
    # Physics-based helper functions
    calculate_ac_power_parameters,
    calculate_ac_efficiency,
    calculate_dc_efficiency,
    # Configuration
    TECHNICAL as TRANSMISSION_TECHNICAL,
)

__all__ = [
    # Shared helpers
    "calculate_mva",
    "calculate_current",
    "calculate_cable_cost_per_km",
    # Coefficients
    "CABLE_COST_COEFFICIENTS",
    "INTRA_ARRAY_TRANSFORMER_COEFFICIENTS",
    "INTER_ARRAY_TRANSFORMER_COEFFICIENTS",
    "SUBSTATION_ANCILLARY_COST",
    # Intra-array
    "calculate_intra_array_cost",
    "calculate_intra_array_cost_for_arrays",
    "calculate_transformer_cost",
    "calculate_string_cable_cost",
    # Inter-array
    "calculate_inter_array_cost",
    "calculate_single_inter_array_cost",
    "calculate_substation_transformer_cost",
    # Transmission (physics-based)
    "calculate_transmission_cost",
    "calculate_hvac_cost",
    "calculate_hvdc_cost",
    "load_ac_cable_data",
    "load_dc_cable_data",
    "calculate_ac_power_parameters",
    "calculate_ac_efficiency",
    "calculate_dc_efficiency",
    "TRANSMISSION_TECHNICAL",
]
