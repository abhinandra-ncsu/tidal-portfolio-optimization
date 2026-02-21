"""
Device Cost Functions
=====================

Functions for calculating tidal turbine device lifecycle costs.

Uses CBS-derived (Cost Breakdown Structure) formulas with 8 CapEx
and 6 OpEx categories, based on the RM1 Reference Model cost analysis.
"""

import numpy as np


# =============================================================================
# CAPEX CATEGORIES
# =============================================================================

def _calculate_development_cost(units):
    """Development costs: Cost = 5,093,988 × units^0.1705"""
    return 5_093_988 * np.power(units, 0.1705)


def _calculate_infrastructure_cost():
    """Infrastructure costs (dedicated O&M vessel): Fixed $14.45M"""
    return 14_450_000


def _calculate_device_structural_cost(units):
    """Device structural: Cost = 982,698 × units^0.9391"""
    return 982_698 * np.power(units, 0.9391)


def _calculate_pto_cost(units):
    """Power Take-Off: Cost = 468,348 + 1,556,123 × units"""
    return 468_348 + 1_556_123 * units


def _calculate_integration_profit_cost(units):
    """Integration & profit margin: Cost = 159,733 + 229,120 × units"""
    return 159_733 + 229_120 * units


def _calculate_installation_cost(units):
    """Installation: Cost = 9,397,592 + 313,210 × units"""
    return 9_397_592 + 313_210 * units


def _calculate_decommissioning_cost(units):
    """Decommissioning: Cost = 9,397,592 + 313,210 × units"""
    return 9_397_592 + 313_210 * units


def _calculate_contingency_cost(units):
    """Contingency: Cost = 3,186,167 + 290,394 × units"""
    return 3_186_167 + 290_394 * units


# =============================================================================
# OPEX CATEGORIES
# =============================================================================

def _calculate_insurance_cost(units):
    """Annual insurance: Cost = 561,046 + 244,134 × log(units)"""
    return 561_046 + 244_134 * np.log(units)


def _calculate_environmental_monitoring_cost(units):
    """Annual environmental monitoring: Cost = 722,176 + 67,426 × log(units)"""
    return 722_176 + 67_426 * np.log(units)


def _calculate_marine_operations_cost(units):
    """Annual marine operations: Cost = 153,325 + 28,968 × units"""
    return 153_325 + 28_968 * units


def _calculate_shoreside_operations_cost(units):
    """Annual shoreside operations: Cost = 250,855 + 2,768 × units"""
    return 250_855 + 2_768 * units


def _calculate_replacement_parts_cost(units):
    """Annual replacement parts: Cost = 11,473 + 31,068 × units"""
    return 11_473 + 31_068 * units


def _calculate_consumables_cost(units):
    """Annual consumables: Cost = 1,874 × units"""
    return 1_874 * units


# =============================================================================
# BREAKDOWNS
# =============================================================================

def calculate_capex_breakdown(num_turbines):
    """
    Calculate detailed CapEx breakdown for a deployment.

    Args:
        num_turbines: Number of turbines

    Returns:
        dict with all CapEx categories and total
    """
    breakdown = {
        'development': _calculate_development_cost(num_turbines),
        'infrastructure': _calculate_infrastructure_cost(),
        'device_structural': _calculate_device_structural_cost(num_turbines),
        'pto': _calculate_pto_cost(num_turbines),
        'integration_profit': _calculate_integration_profit_cost(num_turbines),
        'installation': _calculate_installation_cost(num_turbines),
        'decommissioning': _calculate_decommissioning_cost(num_turbines),
        'contingency': _calculate_contingency_cost(num_turbines),
    }
    breakdown['total'] = sum(breakdown.values())
    return breakdown


def calculate_opex_breakdown(num_turbines):
    """
    Calculate detailed annual OpEx breakdown for a deployment.

    Args:
        num_turbines: Number of turbines

    Returns:
        dict with all OpEx categories and total
    """
    breakdown = {
        'insurance': _calculate_insurance_cost(num_turbines),
        'environmental_monitoring': _calculate_environmental_monitoring_cost(num_turbines),
        'marine_operations': _calculate_marine_operations_cost(num_turbines),
        'shoreside_operations': _calculate_shoreside_operations_cost(num_turbines),
        'replacement_parts': _calculate_replacement_parts_cost(num_turbines),
        'consumables': _calculate_consumables_cost(num_turbines),
    }
    breakdown['total'] = sum(breakdown.values())
    return breakdown


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def calculate_device_cost(num_turbines, fcr):
    """
    Calculate device costs using CBS (Cost Breakdown Structure) model.

    Args:
        num_turbines: Number of turbines
        fcr: Fixed Charge Rate (e.g., 0.113 = 11.3%)

    Returns:
        dict with detailed CapEx/OpEx breakdowns and annualized cost
    """
    capex = calculate_capex_breakdown(num_turbines)
    opex = calculate_opex_breakdown(num_turbines)

    total_capex = capex['total']
    total_opex = opex['total']
    annualized = fcr * total_capex + total_opex

    return {
        'num_turbines': num_turbines,
        'capex_per_turbine': total_capex / num_turbines if num_turbines > 0 else 0,
        'total_capex': total_capex,
        'opex_annual': total_opex,
        'annualized_cost': annualized,
        'fcr': fcr,
        'capex_breakdown': capex,
        'opex_breakdown': opex,
    }
