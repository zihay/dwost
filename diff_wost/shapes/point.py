from dataclasses import dataclass

from diff_wost.core.fwd import Array2, Array3, Int


@dataclass
class Point2D:
    p: Array2
    index: Int  # index in the original array
    sorted_index: Int  # index in the sorted array


@dataclass
class Point3D:
    p: Array3
    index: Int  # index in the original array
    sorted_index: Int  # index in the sorted array
