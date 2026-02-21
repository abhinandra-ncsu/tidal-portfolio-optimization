"""
Wake Loss Functions
===================

Functions for calculating wake effects within tidal turbine arrays.

Uses a constant wake loss factor based on literature values for
typical array spacing (15D streamwise, 4D spanwise, staggered layout).

References:
    - Vennell (2011): Optimal array spacing
    - Divett et al. (2016): Array wake effects
    - González-Gorbeña (2018): Tidal array optimization
"""

import numpy as np


def apply_wake_loss(gross_energy, wake_loss_factor):
    """
    Apply wake loss to gross energy.

    Args:
        gross_energy: Gross energy before wake losses (scalar or array)
        wake_loss_factor: Wake loss multiplier (e.g., 0.88 = 12% loss)

    Returns:
        Net energy after wake losses
    """
    return gross_energy * wake_loss_factor
