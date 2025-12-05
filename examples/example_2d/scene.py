import numpy as np

from assets import ASSET_DIR
from diff_wost.core.distr import DiscreteDistribution
from diff_wost.core.fwd import Array2, Array2i, Float, Int, dr
from diff_wost.render.interaction import BoundarySamplingRecord, ClosestPointRecord
from diff_wost.shapes.polyline import Scene2D
from diff_wost.shapes.silhouette_vertex import SilhouetteVertex
from diff_wost.utils.obj_loader import load_obj_2d


class FlowScene(Scene2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # filter the silhouettes
        p = self.neumann_scene.silhouettes.b
        n0 = self.neumann_scene.silhouettes.n0()
        n1 = self.neumann_scene.silhouettes.n1()
        h0 = self.h(BoundarySamplingRecord(p=p, n=n0))
        h1 = self.h(BoundarySamplingRecord(p=p, n=n1))
        zero_neumann = (dr.abs(h0) < 1e-6) & (dr.abs(h1) < 1e-6)
        self.neumann_scene.silhouettes = dr.gather(
            SilhouetteVertex, self.neumann_scene.silhouettes, dr.compress(~zero_neumann)
        )
        self.neumann_scene.silhouette_pmf = DiscreteDistribution(
            dr.ones(Float, dr.width(self.neumann_scene.silhouettes))
        )

    def u(self, p: Array2):
        return p.x

    def f(self, p: Array2):
        # f = - \nabla u
        return Float(0.0)

    def dfdx(self, p: Array2):
        return Array2(0.0, 0.0)

    @dr.syntax
    def dudx(self, p: Array2):
        ret = Array2(0.0, 0.0)
        if (dr.abs(p.x) > 0.9) | (dr.abs(p.y) > 0.9):
            ret = Array2(1.0, 0.0)
        return ret

    def g(self, c_rec: ClosestPointRecord):
        return Float(0.0)

    def dgdt(self, c_rec: ClosestPointRecord):
        return Float(0.0)

    @dr.syntax
    def h(self, b_rec: BoundarySamplingRecord):
        return dr.dot(self.dudx(b_rec.p), b_rec.n)

    @dr.syntax
    def dhdt(self, b_rec: BoundarySamplingRecord):
        return Float(0.0)


def make_square():
    vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]) * 1.0
    indices = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    return vertices, indices


def make_round_square():
    vertices, indices = load_obj_2d(ASSET_DIR / "rounded_square.obj")
    return vertices, indices


def make_rounded_square_scene() -> FlowScene:
    """
    Create a 2D flow scene with inner and outer boundaries.

    This creates a square outer boundary with a regular polygon inner boundary.

    Args:
        build_bvh: Whether to build a bounding volume hierarchy

    Returns:
        FlowScene object
    """
    # Create outer square boundary
    outer_vertices, outer_indices = make_square()
    outer_types = np.zeros(len(outer_indices), dtype=np.int32)

    inner_vertices, inner_indices = make_round_square()
    inner_vertices = inner_vertices * 0.5
    inner_indices = inner_indices[:, ::-1] + len(outer_vertices)
    inner_types = np.zeros(len(inner_indices), dtype=np.int32)

    # Combine boundaries
    vertices = np.concatenate([outer_vertices, inner_vertices])
    indices = np.concatenate([outer_indices, inner_indices])
    types = np.concatenate([outer_types, inner_types])

    # Create and return scene
    return FlowScene(
        vertices=Array2(vertices.T), indices=Array2i(indices.T), types=Int(types)
    )


def make_square_scene() -> FlowScene:
    """
    Create a 2D flow scene with inner and outer boundaries.

    This creates a square outer boundary with a regular polygon inner boundary.

    Args:
        build_bvh: Whether to build a bounding volume hierarchy

    Returns:
        FlowScene object
    """
    # Create outer square boundary
    outer_vertices, outer_indices = make_square()
    outer_types = np.zeros(len(outer_indices), dtype=np.int32)

    inner_vertices, inner_indices = make_square()
    inner_vertices = inner_vertices * 0.5
    inner_indices = inner_indices[:, ::-1] + len(outer_vertices)
    inner_types = np.zeros(len(inner_indices), dtype=np.int32)

    # Combine boundaries
    vertices = np.concatenate([outer_vertices, inner_vertices])
    indices = np.concatenate([outer_indices, inner_indices])
    types = np.concatenate([outer_types, inner_types])

    # Create and return scene
    return FlowScene(
        vertices=Array2(vertices.T), indices=Array2i(indices.T), types=Int(types)
    )
