"""Core types and utilities for diff_wost."""

from .distr import DiscreteDistribution
from .fwd import (
    BASE_DIR,
    PCG32,
    Array2,
    Array2i,
    Array3,
    Array3i,
    Array4,
    Array4i,
    Bool,
    Epsilon,
    Float,
    Int,
    Matrix2,
    Matrix3,
    Matrix4,
    Quaternion4,
    RayEpsilon,
    Tensor,
    TensorXf,
    TensorXi,
    UInt,
    dr,
)
from .math import (
    closest_point_line_segment,
    cross,
    distance_to_plane,
    in_range,
    is_silhouette,
    mod,
    outer_product,
    rotate90,
    sample_tea_32,
)

__all__ = [
    # drjit
    "dr",
    # Types
    "Float",
    "Int",
    "UInt",
    "Bool",
    "Array2",
    "Array3",
    "Array4",
    "Array2i",
    "Array3i",
    "Array4i",
    "Matrix2",
    "Matrix3",
    "Matrix4",
    "Quaternion4",
    "Tensor",
    "TensorXf",
    "TensorXi",
    "PCG32",
    # Constants
    "Epsilon",
    "RayEpsilon",
    "BASE_DIR",
    # Distribution
    "DiscreteDistribution",
    # Math utilities
    "outer_product",
    "cross",
    "mod",
    "rotate90",
    "in_range",
    "sample_tea_32",
    "distance_to_plane",
    "closest_point_line_segment",
    "is_silhouette",
]
