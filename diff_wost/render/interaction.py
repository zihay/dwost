"""Interaction records for geometric queries and sampling.

This module defines dataclasses that store results of geometric operations
like ray intersections, closest point queries, and boundary sampling.
These records are used throughout the Walk on Stars implementation.
"""

from dataclasses import dataclass

from diff_wost.core.fwd import Array2, Array3, Bool, Float, Int, dr


@dataclass
class Intersection:
    """Record of a 2D ray-geometry intersection.

    Attributes:
        valid: Whether the intersection is valid.
        p: Intersection point.
        n: Surface normal at intersection.
        t: Parameter along the intersected primitive.
        d: Distance from ray origin to intersection.
        prim_id: Index of the intersected primitive.
        on_boundary: Whether the point is on a boundary.
        type: Boundary type (Dirichlet=1, Neumann=0, or -1 for unset).
    """

    valid: Bool
    p: Array2
    n: Array2
    t: Float
    d: Float
    prim_id: Int
    on_boundary: Bool
    type: Int = Int(-1)


@dataclass
class Intersection3D:
    """Record of a 3D ray-geometry intersection.

    Attributes:
        valid: Whether the intersection is valid.
        p: Intersection point.
        n: Surface normal at intersection.
        uv: Barycentric coordinates on the triangle.
        d: Distance from ray origin to intersection.
        prim_id: Index of the intersected primitive.
        on_boundary: Whether the point is on a boundary.
        type: Boundary type.
    """

    valid: Bool
    p: Array3
    n: Array3
    uv: Array2
    d: Float
    prim_id: Int
    on_boundary: Bool
    type: Int = Int(-1)


@dataclass
class ClosestPointRecord:
    """Record of a 2D closest point query.

    Attributes:
        p: Closest point on the geometry.
        n: Surface normal at the closest point.
        t: Parameter along the primitive (for edges: 0 to 1).
        d: Distance to the closest point.
        prim_id: Index of the closest primitive.
        type: Boundary type.
        J: Jacobian factor for gradient computation.
        valid: Whether the record contains valid data.
    """

    p: Array2
    n: Array2
    t: Float
    d: Float = Float(dr.inf)
    prim_id: Int = Int(-1)
    type: Int = Int(-1)
    J: Float = Float(1.0)
    valid: Bool = Bool(False)


@dataclass
class ClosestPointRecord3D:
    """Record of a 3D closest point query.

    Attributes:
        p: Closest point on the geometry.
        n: Surface normal at the closest point.
        uv: Barycentric coordinates (for triangles).
        d: Distance to the closest point.
        prim_id: Index of the closest primitive.
        type: Boundary type.
        valid: Whether the record contains valid data.
    """

    p: Array3
    n: Array3
    uv: Array2
    d: Float = Float(dr.inf)
    prim_id: Int = Int(-1)
    type: Int = Int(-1)
    valid: Bool = Bool(False)


@dataclass
class BoundarySamplingRecord:
    """Record of a 2D boundary sampling result.

    Attributes:
        p: Sampled point on the boundary.
        n: Surface normal at the sampled point.
        t: Parameter along the primitive.
        pdf: Probability density of the sample.
        prim_id: Index of the sampled primitive.
        type: Boundary type.
        pmf: Probability mass function value (for discrete selection).
    """

    p: Array2
    n: Array2
    t: Float = Float(0.0)
    pdf: Float = Float(1.0)
    prim_id: Int = Int(-1)
    type: Int = Int(-1)
    pmf: Float = Float(1.0)


@dataclass
class BoundarySamplingRecord3D:
    """Record of a 3D boundary sampling result.

    Attributes:
        p: Sampled point on the boundary.
        n: Surface normal at the sampled point.
        uv: Parametric coordinates on the triangle.
        pdf: Probability density of the sample.
        prim_id: Index of the sampled primitive.
        pmf: Probability mass function value.
        type: Boundary type.
    """

    p: Array3
    n: Array3
    uv: Array2 = Array2(0.0, 0.0)
    pdf: Float = Float(1.0)
    prim_id: Int = Int(-1)
    pmf: Float = Float(1.0)
    type: Int = Int(-1)


@dataclass
class SilhouetteSamplingRecord:
    """Record of a 2D silhouette vertex sampling result.

    Silhouette vertices are where the boundary changes orientation
    relative to the query point. They require special handling in WoSt.

    Attributes:
        p: Position of the silhouette vertex.
        n1: Normal of the first adjacent edge.
        n2: Normal of the second adjacent edge.
        t1: Tangent of the first adjacent edge.
        t2: Tangent of the second adjacent edge.
        pdf: Sampling probability density.
        T1: Type of first adjacent primitive.
        T2: Type of second adjacent primitive.
        prim_id: Index of the silhouette vertex.
    """

    p: Array2
    n1: Array2
    n2: Array2
    t1: Array2
    t2: Array2
    pdf: Float
    T1: Int = Int(-1)
    T2: Int = Int(-1)
    prim_id: Int = Int(-1)


@dataclass
class SilhouetteSamplingRecord3D:
    """Record of a 3D silhouette edge sampling result.

    In 3D, silhouettes are edges where adjacent faces change visibility.

    Attributes:
        p: Point on the silhouette edge.
        n1: Normal of the first adjacent face.
        n2: Normal of the second adjacent face.
        t1: Tangent direction from first face.
        t2: Tangent direction from second face.
        pdf: Sampling probability density.
        T1: Type of first adjacent primitive.
        T2: Type of second adjacent primitive.
        prim_id: Index of the silhouette edge.
    """

    p: Array3
    n1: Array3
    n2: Array3
    t1: Array3
    t2: Array3
    pdf: Float
    T1: Int = Int(-1)
    T2: Int = Int(-1)
    prim_id: Int = Int(-1)


@dataclass
class ClosestSilhouettePointRecord:
    """Record of a 2D closest silhouette point query.

    Attributes:
        valid: Whether a silhouette point was found.
        p: Position of the closest silhouette point.
        d: Distance to the silhouette point.
        prim_id: Index of the silhouette vertex.
    """

    valid: Bool
    p: Array2
    d: Float = Float(dr.inf)
    prim_id: Int = Int(-1)


@dataclass
class ClosestSilhouettePointRecord3D:
    """Record of a 3D closest silhouette point query.

    Attributes:
        valid: Whether a silhouette point was found.
        p: Position of the closest silhouette point.
        d: Distance to the silhouette point.
        prim_id: Index of the silhouette edge.
    """

    valid: Bool
    p: Array3
    d: Float = Float(dr.inf)
    prim_id: Int = Int(-1)
