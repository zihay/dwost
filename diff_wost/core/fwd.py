from pathlib import Path

import drjit as dr
import numpy as np
from drjit.auto.ad import PCG32, Bool, TensorXf, TensorXi
from drjit.auto.ad import Array2f as Array2
from drjit.auto.ad import Array2i as Array2i
from drjit.auto.ad import Array3f as Array3
from drjit.auto.ad import Array3i as Array3i
from drjit.auto.ad import Array4f as Array4
from drjit.auto.ad import Array4i as Array4i
from drjit.auto.ad import Float32 as Float
from drjit.auto.ad import Int32 as Int
from drjit.auto.ad import Matrix2f as Matrix2
from drjit.auto.ad import Matrix3f as Matrix3
from drjit.auto.ad import Matrix4f as Matrix4
from drjit.auto.ad import Quaternion4f as Quaternion4
from drjit.auto.ad import TensorXf as Tensor
from drjit.auto.ad import UInt32 as UInt

BASE_DIR = Path(__file__).parent.parent.parent

Epsilon = 1e-3
RayEpsilon = 1e-5

__all__ = [
    "dr",
    "np",
    "PCG32",
    "Bool",
    "TensorXf",
    "TensorXi",
    "Tensor",
    "Array2",
    "Array2i",
    "Array3",
    "Array3i",
    "Array4",
    "Array4i",
    "Float",
    "Int",
    "UInt",
    "Matrix2",
    "Matrix3",
    "Matrix4",
    "Quaternion4",
    "BASE_DIR",
    "Epsilon",
    "RayEpsilon",
]
