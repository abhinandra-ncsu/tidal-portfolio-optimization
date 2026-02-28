"""
Intra-Array Cable and Transformer Cost Functions
=================================================

Cabling and transformer costs within a turbine array using
Collin et al. (2017) empirical cost models.

Methodology: see writing/electrical-infrastructure/intra-array-cables.md
"""

import numpy as np

from .coefficients import INTRA_ARRAY_TRANSFORMER_COEFFICIENTS
from .helpers import calculate_mva, calculate_current, calculate_cable_cost_per_km


# =============================================================================
# VOLTAGE SELECTION
# =============================================================================

def calculate_minimum_voltage(max_mva, max_current_a):
    """
    Calculate minimum voltage required to keep current under limit.

    Args:
        max_mva: Maximum MVA to be transmitted
        max_current_a: Maximum allowable current in amperes

    Returns:
        Minimum voltage in volts
    """
    return (max_mva * 1e6) / (np.sqrt(3) * max_current_a)


def select_standard_voltage(min_voltage_v):
    """
    Select the next standard voltage level above the minimum required.

    Args:
        min_voltage_v: Minimum voltage required in volts

    Returns:
        Standard voltage level in volts
    """
    standard_voltages = [6600, 11000, 22000, 33000, 66000]
    for v in standard_voltages:
        if v >= min_voltage_v:
            return v
    return standard_voltages[-1]


# =============================================================================
# COST CALCULATION FUNCTIONS
# =============================================================================

def calculate_transformer_cost(mva, transformer_type='LV_MV_DRY'):
    """
    Calculate transformer cost using Collin 2017 power model.

    Model: cost ($) = c1 * S^c2 + c3, where S = MVA rating

    Args:
        mva: Transformer MVA rating
        transformer_type: Type of transformer (default: 'LV_MV_DRY')

    Returns:
        Transformer cost in $
    """
    if transformer_type not in INTRA_ARRAY_TRANSFORMER_COEFFICIENTS:
        raise ValueError(f"Unknown transformer type: {transformer_type}. "
                        f"Available: {list(INTRA_ARRAY_TRANSFORMER_COEFFICIENTS.keys())}")

    coeffs = INTRA_ARRAY_TRANSFORMER_COEFFICIENTS[transformer_type]
    c1, c2, c3 = coeffs['c1'], coeffs['c2'], coeffs['c3']

    return c1 * (mva ** c2) + c3


def calculate_string_cable_cost(num_turbines, turbine_power_mw,
                                 spacing_km, voltage_v, power_factor):
    """
    Calculate total cable cost for one string of turbines.

    For N turbines there are N-1 cable segments between them (daisy chain).
    Segment i carries accumulated power from (i+1) upstream turbines.
    Cost is calculated segment-by-segment using the Collin 2017 model.

    Args:
        num_turbines: Number of turbines in the string
        turbine_power_mw: Power per turbine in MW
        spacing_km: Spacing between turbines in km
        voltage_v: Cable voltage in volts
        power_factor: Power factor

    Returns:
        dict with total cost and segment breakdown
    """
    segments = []
    total_cost = 0.0
    total_length = 0.0

    # N turbines → N-1 cable segments between them
    for i in range(num_turbines - 1):
        num_upstream = i + 1
        power_mw = num_upstream * turbine_power_mw
        mva = calculate_mva(power_mw, power_factor)
        current_a = calculate_current(mva, voltage_v)
        cost_per_km = calculate_cable_cost_per_km(mva, voltage_v)
        segment_cost = cost_per_km * spacing_km

        segments.append({
            'segment_index': i,
            'num_turbines_upstream': num_upstream,
            'power_mw': power_mw,
            'mva': mva,
            'current_a': current_a,
            'length_km': spacing_km,
            'cost_per_km': cost_per_km,
            'segment_cost': segment_cost,
        })

        total_cost += segment_cost
        total_length += spacing_km

    return {
        'num_turbines': num_turbines,
        'total_length_km': total_length,
        'total_cost': total_cost,
        'voltage_v': voltage_v,
        'segments': segments,
    }


# =============================================================================
# MAIN COST FUNCTIONS
# =============================================================================

def calculate_intra_array_cost(rows, cols, turbine_power_mw,
                                power_factor, row_spacing_km,
                                col_spacing_km, voltage_v, fcr):
    """
    Calculate total intra-array cable and transformer costs for one array.

    Uses Collin et al. (2017) empirical cost models for:
    - Submarine cables at specified voltage level
    - LV (690V) to MV transformers at each turbine

    Array topology:
    - Grid layout with 'rows' horizontal strings of 'cols' turbines each
    - Each turbine has a 690V to MV step-up transformer
    - N turbines per string → N-1 daisy-chain cable segments
    - Collection point at the middle string (minimizes row cable length)
    - Row cables connect each string's end to the collection point

    Args:
        rows: Number of rows/strings in array
        cols: Number of turbines per string
        turbine_power_mw: Power per turbine in MW
        power_factor: Power factor
        row_spacing_km: Spacing between rows in km
        col_spacing_km: Spacing between columns in km
        voltage_v: Intra-array voltage in volts
        fcr: Fixed Charge Rate for annualization

    Returns:
        dict with cable costs, transformer costs, and totals
    """
    num_turbines = rows * cols
    turbine_mva = calculate_mva(turbine_power_mw, power_factor)

    # -------------------------------------------------------------------------
    # Cable costs: segment-by-segment within strings + row connections
    # -------------------------------------------------------------------------

    # String cables (within each row)
    string_result = calculate_string_cable_cost(
        num_turbines=cols,
        turbine_power_mw=turbine_power_mw,
        spacing_km=col_spacing_km,
        voltage_v=voltage_v,
        power_factor=power_factor,
    )
    string_cable_cost = string_result['total_cost'] * rows
    string_cable_length = string_result['total_length_km'] * rows

    # Row connection cables (connecting strings to collector)
    # Collection point is at the middle string to minimize total cable length.
    # Each row cable carries the full string power.
    string_mva = calculate_mva(cols * turbine_power_mw, power_factor)
    row_cable_cost_per_km = calculate_cable_cost_per_km(string_mva, voltage_v)

    # Each string's distance to the middle string: |i - middle| * row_spacing
    middle = rows // 2
    row_cable_length = sum(
        abs(i - middle) * row_spacing_km for i in range(rows)
    )
    row_cable_cost = row_cable_length * row_cable_cost_per_km

    total_cable_length = string_cable_length + row_cable_length
    total_cable_cost = string_cable_cost + row_cable_cost

    # -------------------------------------------------------------------------
    # Transformer costs: one per turbine
    # -------------------------------------------------------------------------

    transformer_cost_each = calculate_transformer_cost(turbine_mva, 'LV_MV_DRY')
    total_transformer_cost = transformer_cost_each * num_turbines

    # -------------------------------------------------------------------------
    # Totals and annualization
    # -------------------------------------------------------------------------

    total_capex = total_cable_cost + total_transformer_cost
    annualized_cost = total_capex * fcr

    return {
        # Cable breakdown
        'cable': {
            'total_length_km': total_cable_length,
            'string_cable_length_km': string_cable_length,
            'row_cable_length_km': row_cable_length,
            'total_capex': total_cable_cost,
            'string_cable_cost': string_cable_cost,
            'row_cable_cost': row_cable_cost,
            'voltage_v': voltage_v,
            'string_segments': string_result['segments'],
        },
        # Transformer breakdown
        'transformer': {
            'num_transformers': num_turbines,
            'mva_each': turbine_mva,
            'cost_each': transformer_cost_each,
            'total_capex': total_transformer_cost,
            'type': 'LV_MV_DRY',
        },
        # Totals
        'total_capex': total_capex,
        'annualized_cost': annualized_cost,
        'fcr': fcr,
        # Array info
        'layout': {
            'rows': rows,
            'cols': cols,
            'num_turbines': num_turbines,
            'turbine_power_mw': turbine_power_mw,
            'power_factor': power_factor,
            'row_spacing_km': row_spacing_km,
            'col_spacing_km': col_spacing_km,
        },
        # Methodology documentation
        'methodology': {
            'cable_model': 'Collin et al. (2017) Table A1',
            'transformer_model': 'Collin et al. (2017) Table A3',
            'voltage_selection': (
                f'{voltage_v/1000:.0f} kV selected based on: '
                f'(1) max string power = {cols * turbine_power_mw:.1f} MW = '
                f'{calculate_mva(cols * turbine_power_mw, power_factor):.2f} MVA, '
                f'(2) current limit 500 A requires >= '
                f'{calculate_minimum_voltage(calculate_mva(cols * turbine_power_mw, power_factor), 500)/1000:.1f} kV, '
                f'(3) cost optimization favors lower voltage for arrays < 50 MW'
            ),
            'transformer_architecture': 'One LV to MV transformer per turbine',
        },
    }


def calculate_intra_array_cost_for_arrays(num_arrays, rows, cols,
                                           turbine_power_mw, power_factor,
                                           row_spacing_km, col_spacing_km,
                                           voltage_v, fcr):
    """
    Calculate total intra-array costs for multiple arrays.

    Args:
        num_arrays: Number of arrays
        rows: Number of rows/strings per array
        cols: Number of turbines per string
        turbine_power_mw: Power per turbine in MW
        power_factor: Power factor
        row_spacing_km: Spacing between rows in km
        col_spacing_km: Spacing between columns in km
        voltage_v: Intra-array voltage in volts
        fcr: Fixed Charge Rate for annualization

    Returns:
        dict with total costs for all arrays
    """
    single = calculate_intra_array_cost(
        rows=rows, cols=cols,
        turbine_power_mw=turbine_power_mw,
        power_factor=power_factor,
        row_spacing_km=row_spacing_km,
        col_spacing_km=col_spacing_km,
        voltage_v=voltage_v,
        fcr=fcr,
    )

    return {
        'num_arrays': num_arrays,
        # Cable totals
        'cable': {
            'total_length_km': single['cable']['total_length_km'] * num_arrays,
            'total_capex': single['cable']['total_capex'] * num_arrays,
            'voltage_v': single['cable']['voltage_v'],
        },
        # Transformer totals
        'transformer': {
            'num_transformers': single['transformer']['num_transformers'] * num_arrays,
            'total_capex': single['transformer']['total_capex'] * num_arrays,
        },
        # Overall totals
        'total_capex': single['total_capex'] * num_arrays,
        'annualized_cost': single['annualized_cost'] * num_arrays,
        'fcr': single['fcr'],
        # Per-array info
        'per_array': single,
    }
