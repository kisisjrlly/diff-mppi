"""
Diff-MPPI: Differentiable Model Predictive Path Integral Control Library

A clean implementation of Path Integral Networks and their accelerated variants.
"""

from .core import DiffMPPI, create_mppi_controller

__version__ = "1.0.0"
__all__ = ["DiffMPPI", "create_mppi_controller"]
