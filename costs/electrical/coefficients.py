"""
Collin et al. (2017) Empirical Cost Coefficients
=================================================

Unified cable and transformer cost coefficients from Collin et al. (2017).

All cable cost models use the exponential form:
    Cost (kEUR/km) = c1 + c2 * exp(c3 * S), where S = MVA rating

Transformer models vary:
    Intra-array (power model): Cost ($) = c1 * S^c2 + c3
    Inter-array (linear model): Cost ($) = c1 * S + c3
"""

# Currency conversion (EUR to USD)
EUR_TO_USD = 1.1

# =============================================================================
# CABLE COST COEFFICIENTS — Collin et al. (2017) Table A1
# =============================================================================
# Unified dict covering intra-array (6.6-33 kV) and inter-array (33-132 kV)
# voltages. The 33 kV entry is identical in both original files.
CABLE_COST_COEFFICIENTS = {
    # Intra-array voltages
    6600: {
        'c1': 67.63, 'c2': 8.24, 'c3': 0.44,
        'mva_range': (2.9, 7.5),
        'description': '6.6 kV submarine cable',
    },
    11000: {
        'c1': 49.37, 'c2': 16.32, 'c3': 0.22,
        'mva_range': (4.8, 12.5),
        'description': '11 kV submarine cable',
    },
    22000: {
        'c1': -1.27, 'c2': 50.66, 'c3': 0.07,
        'mva_range': (9.5, 27.2),
        'description': '22 kV submarine cable',
    },
    # Shared voltage (intra-array and inter-array)
    33000: {
        'c1': -35.29, 'c2': 80.17, 'c3': 0.04,
        'mva_range': (17.0, 44.0),
        'description': '33 kV submarine cable',
    },
    # Inter-array voltages
    66000: {
        'c1': -57.35, 'c2': 105.20, 'c3': 0.02,
        'mva_range': (34.3, 94.3),
        'description': '66 kV submarine cable',
    },
    132000: {
        'c1': -1337.00, 'c2': 1125.00, 'c3': 0.0035,
        'mva_range': (121.1, 188.6),
        'description': '132 kV submarine cable',
    },
}

# =============================================================================
# TRANSFORMER COST COEFFICIENTS — Collin et al. (2017) Table A3
# =============================================================================

# Intra-array: LV (690V) to MV step-up transformers (one per turbine)
# Power model: Cost ($) = c1 * S^c2 + c3
INTRA_ARRAY_TRANSFORMER_COEFFICIENTS = {
    'LV_MV_DRY': {
        'c1': 64960, 'c2': 0.6329, 'c3': 7307,
        'mva_range': (0.1125, 3.0),
        'description': 'LV (690V) to MV (up to 33kV) dry-type transformer',
    },
    'LV_MV_OIL': {
        'c1': 454800, 'c2': 0.6329, 'c3': 51115,
        'mva_range': (0.1125, 3.0),
        'description': 'LV (690V) to MV (up to 33kV) oil-type transformer',
    },
}

# Inter-array: MV to HV step-up transformers (one per array substation)
# Linear model: Cost ($) = c1 * S + c3 (c2 = 1)
INTER_ARRAY_TRANSFORMER_COEFFICIENTS = {
    'MV_33kV': {
        'c1': 26290, 'c2': 1, 'c3': 0,
        'mva_range': (5.0, 100.0),
        'description': '11 kV to 33 kV step-up transformer',
    },
    'MV_66kV': {
        'c1': 39435, 'c2': 1, 'c3': 0,
        'mva_range': (5.0, 100.0),
        'description': '11 kV to 66 kV step-up transformer',
    },
    'MV_132kV': {
        'c1': 52580, 'c2': 1, 'c3': 0,
        'mva_range': (30.0, 150.0),
        'description': '11/33 kV to 132 kV step-up transformer',
    },
}

# =============================================================================
# SUBSTATION ANCILLARY COSTS
# =============================================================================
# Subsea enclosure/platform, switchgear, protection systems, and installation
# per array substation.
# Refs: Smart & Noonan (2016) Table 4.2; Nakhai et al. (2023)
SUBSTATION_ANCILLARY_COST = 3_500_000  # $/substation
