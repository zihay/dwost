"""Primitive types and enumerations for boundary conditions.

This module defines fundamental types used in the Walk on Stars
implementation for handling different boundary condition types.
"""

from enum import Enum


class BoundaryType(Enum):
    """Enumeration of supported boundary condition types.

    Attributes:
        Neumann: Neumann boundary condition (∂u/∂n = h specified).
        Dirichlet: Dirichlet boundary condition (u = g specified).
    """

    Neumann = 0
    Dirichlet = 1
