"""
Diff-MPPI: Differentiable Model Predictive Path Integral Control Library

A clean implementation of Path Integral Networks and their accelerated variants,
now with Latent Space Model Predictive Path Integral (LMPPI) support.
"""

from .core import DiffMPPI, create_mppi_controller

__version__ = "1.0.0"
__all__ = ["DiffMPPI", "create_mppi_controller"]

# Optional LMPPI imports - only available if user needs them
try:
    from . import lmppi
    __all__.append("lmppi")
except ImportError:
    # LMPPI dependencies may not be available
    pass
