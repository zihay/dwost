"""
3D Flow Scene Module

This module provides scene definitions for 3D flow visualization.
"""

import numpy as np

from assets import ASSET_DIR
from diff_wost.core.distr import DiscreteDistribution
from diff_wost.core.fwd import Array3, Array3i, Float, Int, Matrix3, dr
from diff_wost.shapes.mesh import Scene3D
from diff_wost.shapes.silhouette_edge import SilhouetteEdge
from diff_wost.utils.obj_loader import load_obj_3d


class TestScene3D(Scene3D):
    """Base class for 3D flow scenes with Laplace equation boundary conditions."""

    def __init__(
        self,
        vertices: Array3,
        indices: Array3i,
        types: Int,
        box_min: Array3,
        box_max: Array3,
    ):
        super().__init__(vertices, indices, types)
        self.box_min = box_min
        self.box_max = box_max
        # filter the silhouettes
        a = self.neumann_scene.silhouettes.a
        b = self.neumann_scene.silhouettes.b
        ga = self.dudx(a)
        gb = self.dudx(b)
        zero_neumann = (dr.norm(ga) < 1e-6) & (dr.norm(gb) < 1e-6)
        self.neumann_scene.silhouettes = dr.gather(
            SilhouetteEdge, self.neumann_scene.silhouettes, dr.compress(~zero_neumann)
        )
        self.neumann_scene.silhouette_pmf = DiscreteDistribution(
            dr.ones(Float, dr.width(self.neumann_scene.silhouettes))
        )

    def u(self, p: Array3) -> Float:
        """Boundary value function (Dirichlet condition)."""
        return p.x

    def f(self, p: Array3) -> Float:
        """Source term (zero for Laplace equation)."""
        return Float(0.0)

    def dfdx(self, p: Array3) -> Array3:
        """Gradient of source term."""
        return Array3(0.0, 0.0, 0.0)

    @dr.syntax
    def dudx(self, p: Array3) -> Array3:
        ret = Array3(0.0, 0.0, 0.0)
        if (
            (p.x > self.box_min.x + 1e-2)
            & (p.x < self.box_max.x - 1e-2)
            & (p.y > self.box_min.y + 1e-2)
            & (p.y < self.box_max.y - 1e-2)
            & (p.z > self.box_min.z + 1e-2)
            & (p.z < self.box_max.z - 1e-2)
        ):  # inside the box
            ret = Array3(0.0, 0.0, 0.0)
        else:
            ret = Array3(1.0, 0.0, 0.0)
        return ret

    def hessian(self, p: Array3) -> Matrix3:
        """Hessian of solution (zero for linear solution)."""
        return Matrix3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def make_test_scene() -> TestScene3D:
    """
    Create a 3D flow scene with dolphin inside a box.

    Returns:
        TestScene3D object with dolphin mesh inside a bounding box
    """
    # Load outer box
    outer_vertices, outer_indices = load_obj_3d(ASSET_DIR / "whale_box.obj")

    # Load inner dolphin mesh
    inner_vertices, inner_indices = load_obj_3d(ASSET_DIR / "whale.obj")

    # Reverse winding for inner mesh and offset indices
    inner_indices = inner_indices[:, ::-1] + len(outer_vertices)

    # Combine meshes
    vertices = np.concatenate([outer_vertices, inner_vertices])
    indices = np.concatenate([outer_indices, inner_indices])
    types = np.zeros(len(indices), dtype=np.int32)

    return TestScene3D(
        vertices=Array3(vertices.T),
        indices=Array3i(indices.T),
        types=Int(types),
        box_min=Array3(outer_vertices.min(axis=0)),
        box_max=Array3(outer_vertices.max(axis=0)),
    )


if __name__ == "__main__":
    from examples.example_3d.run import read_points

    scene = make_test_scene()
    points = read_points("whale_points.obj")

    # Test inside function
    pts = Array3(points.T)
    inside = scene.inside(pts)
    print(f"Points inside: {dr.sum(Int(inside))} / {dr.width(pts)}")
