"""Mathematical utilities for geometry and sampling.

This module provides fundamental mathematical operations used throughout
the Walk on Stars implementation, including 2D/3D geometry, random sampling,
and geometric queries.
"""

from diff_wost.core.fwd import (
    PCG32,
    Array2,
    Array3,
    Bool,
    Float,
    Int,
    Matrix2,
    UInt,
    dr,
    np,
)


def outer_product(a: Array2, b: Array2) -> Matrix2:
    """Compute the outer product of two 2D vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        2x2 matrix representing a ⊗ b.
    """
    return Matrix2(a.x * b.x, a.x * b.y, a.y * b.x, a.y * b.y)


def cross(u: Array2, v: Array2) -> Float:
    """Compute the 2D cross product (z-component of 3D cross product).

    Args:
        u: First vector.
        v: Second vector.

    Returns:
        Scalar value u × v = u.x * v.y - u.y * v.x.
    """
    return u[0] * v[1] - u[1] * v[0]


def mod(x: Float, y: Float) -> Float:
    """Compute the floating-point modulo operation.

    Args:
        x: Dividend.
        y: Divisor.

    Returns:
        x mod y, always non-negative for positive y.
    """
    return x - y * dr.floor(x / y)


def rotate90(u: Array2) -> Array2:
    """Rotate a 2D vector by 90 degrees counter-clockwise.

    Args:
        u: Input vector.

    Returns:
        Rotated vector (-u.y, u.x).
    """
    return Array2(-u[1], u[0])


def in_range(x: Float, a: Float, b: Float) -> Bool:
    """Check if a value is within a closed interval.

    Args:
        x: Value to test.
        a: Lower bound.
        b: Upper bound.

    Returns:
        True if a <= x <= b.
    """
    return (x >= a) & (x <= b)


def sample_tea_32(v0: UInt, v1: UInt, rounds: int = 4) -> tuple:
    """Generate decorrelated random seeds using TEA (Tiny Encryption Algorithm).

    TEA is used to hash sample indices into uncorrelated initial states for
    the PCG32 random number generator, ensuring independent random streams.

    Args:
        v0: First input value (typically seed).
        v1: Second input value (typically sample index).
        rounds: Number of TEA rounds (default 4 for good mixing).

    Returns:
        Tuple of two hashed 32-bit unsigned integers.
    """
    v0, v1 = UInt(v0), UInt(v1)
    s = UInt(0)
    for _ in range(rounds):
        s += UInt(0x9E3779B9)  # Golden ratio constant
        v0 += ((v1 << 4) + UInt(0xA341316C)) ^ (v1 + s) ^ ((v1 >> 5) + UInt(0xC8013EA4))
        v1 += ((v0 << 4) + UInt(0xAD90777D)) ^ (v0 + s) ^ ((v0 >> 5) + UInt(0x7E95761E))
    return v0, v1


def distance_to_plane(x: Array2, a: Array2, n: Array2) -> Float:
    """Compute signed distance from a point to a plane (2D line).

    Args:
        x: Query point.
        a: Point on the plane/line.
        n: Normal vector of the plane/line.

    Returns:
        Signed distance (positive on normal side, negative on opposite side).
    """
    return dr.dot(x - a, n)


def closest_point_line_segment(x: Array2, a: Array2, b: Array2) -> tuple:
    """Find the closest point on a line segment to a query point.

    Args:
        x: Query point.
        a: First endpoint of the segment.
        b: Second endpoint of the segment.

    Returns:
        Tuple of (distance, closest_point, parameter_t) where t ∈ [0, 1]
        interpolates between a (t=0) and b (t=1).
    """
    u = b - a
    v = x - a
    t = dr.clamp(dr.dot(v, u) / dr.dot(u, u), 0.0, 1.0)
    p = dr.lerp(a, b, t)
    d = dr.norm(p - x)
    return d, p, t


def is_silhouette(x: Array2, a: Array2, b: Array2, c: Array2) -> Bool:
    """Check if vertex b is a silhouette vertex when viewed from x.

    A vertex is a silhouette if the two adjacent edges have opposite
    orientations when viewed from the query point.

    Args:
        x: Query/viewing point.
        a: Previous vertex in the polygon.
        b: Current vertex (potential silhouette).
        c: Next vertex in the polygon.

    Returns:
        True if b is a silhouette vertex.
    """
    return cross(b - a, x - a) * cross(c - b, x - b) < 0.0


@dr.syntax
def ray_intersection(x: Array2, v: Array2, a: Array2, b: Array2) -> Float:
    u = b - a
    w = x - a
    d = cross(v, u)
    s = cross(v, w) / d
    t = cross(u, w) / d
    valid = (t > 0.0) & (s >= 0.0) & (s <= 1.0)
    return dr.select(valid, t, dr.inf)


@dr.syntax
def ray_triangle_intersect(
    x: Array3, d: Array3, a: Array3, b: Array3, c: Array3, r_max: Float = Float(dr.inf)
):
    from diff_wost.render.interaction import Intersection3D

    its = dr.zeros(Intersection3D)
    v1 = b - a
    v2 = c - a
    p = dr.cross(d, v2)
    det = dr.dot(v1, p)
    if dr.abs(det) > dr.epsilon(Float):
        inv_det = 1.0 / det
        s = x - a
        v = dr.dot(s, p) * inv_det
        if (v >= 0) & (v <= 1):
            q = dr.cross(s, v1)
            w = dr.dot(d, q) * inv_det
            if (w >= 0) & (v + w <= 1):
                t = dr.dot(v2, q) * inv_det
                if (t >= 0) & (t <= r_max):
                    its = Intersection3D(
                        valid=Bool(True),
                        p=a + v1 * v + v2 * w,
                        n=dr.normalize(dr.cross(v1, v2)),
                        uv=Array2(1.0 - v - w, v),
                        d=t,
                        prim_id=Int(-1),
                        on_boundary=Bool(True),
                        type=Int(-1),
                    )
    return its


def closest_point_triangle(p: Array3, a: Array3, b: Array3, c: Array3):
    pt = Array3(0, 0, 0)
    uv = Array2(0, 0)
    d = dr.inf
    ab = b - a
    ac = c - a
    active = Bool(True)
    # check if p is in the vertex region outside a
    ax = p - a
    d1 = dr.dot(ab, ax)
    d2 = dr.dot(ac, ax)
    cond = (d1 <= 0) & (d2 <= 0)
    pt = dr.select(cond, a, pt)
    uv = dr.select(cond, Array2(1, 0), uv)
    d = dr.select(cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the vertex region outside b
    bx = p - b
    d3 = dr.dot(ab, bx)
    d4 = dr.dot(ac, bx)
    cond = (d3 >= 0) & (d4 <= d3)
    pt = dr.select(active & cond, b, pt)
    uv = dr.select(active & cond, Array2(0, 1), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the vertex region outside c
    cx = p - c
    d5 = dr.dot(ab, cx)
    d6 = dr.dot(ac, cx)
    cond = (d6 >= 0) & (d5 <= d6)
    pt = dr.select(active & cond, c, pt)
    uv = dr.select(active & cond, Array2(0, 0), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the edge region of ab, if so return projection of p onto ab
    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    cond = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    pt = dr.select(active & cond, a + ab * v, pt)
    uv = dr.select(active & cond, Array2(1 - v, v), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the edge region of ac, if so return projection of p onto ac
    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    cond = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    pt = dr.select(active & cond, a + ac * w, pt)
    uv = dr.select(active & cond, Array2(1 - w, 0), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is in the edge region of bc, if so return projection of p onto bc
    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    cond = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    pt = dr.select(active & cond, b + (c - b) * w, pt)
    uv = dr.select(active & cond, Array2(0, 1 - w), uv)
    d = dr.select(active & cond, dr.norm(p - pt), d)
    active = active & ~cond
    # check if p is inside face region
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    pt = dr.select(active, a + ab * v + ac * w, pt)
    uv = dr.select(active, Array2(1 - v - w, v), uv)
    d = dr.select(active, dr.norm(p - pt), d)
    return pt, Array2(uv[1], 1.0 - uv[0] - uv[1]), d


def uniform_angle(sampler: PCG32) -> Float:
    theta = Float(sampler.next_float32()) * 2.0 * np.pi
    return theta


def uniform_on_circle(sampler: PCG32) -> Array2:
    theta = 2.0 * dr.pi * sampler.next_float32()
    return Array2(dr.cos(theta), dr.sin(theta))


def uniform_in_disk(sampler: PCG32) -> Array2:
    r = dr.sqrt(sampler.next_float32())
    theta = 2.0 * dr.pi * sampler.next_float32()
    return Array2(r * dr.cos(theta), r * dr.sin(theta))


def uniform_in_ball(sampler: PCG32) -> Array3:
    r = dr.sqrt(sampler.next_float32())
    theta = 2.0 * dr.pi * sampler.next_float32()
    z = 1.0 - 2.0 * sampler.next_float32()
    return Array3(r * dr.cos(theta), r * dr.sin(theta), z)


def uniform_on_sphere(sampler: PCG32) -> Array3:
    u = Array2(sampler.next_float32(), sampler.next_float32())
    z = 1.0 - 2.0 * u[0]
    r = dr.sqrt(dr.maximum(0.0, 1.0 - z * z))
    theta = 2.0 * dr.pi * u[1]
    return Array3(r * dr.cos(theta), r * dr.sin(theta), z)


def interpolate(a, b, c, uv):
    return b * uv[0] + c * uv[1] + a * (1 - uv[0] - uv[1])


@dr.syntax
def inside_triangle(p: Array3, a: Array3, b: Array3, c: Array3):
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = dr.dot(v0, v0)
    dot01 = dr.dot(v0, v1)
    dot02 = dr.dot(v0, v2)
    dot11 = dr.dot(v1, v1)
    dot12 = dr.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    is_inside = Bool(True)
    if denom == 0:
        is_inside = Bool(False)  # Triangle is degenerate
    else:
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom

        is_inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    return is_inside


def concat(a, b):
    """Concatenate two Dr.Jit arrays of the same type.

    Args:
        a: First array.
        b: Second array (must be same type as a).

    Returns:
        Concatenated array [a, b].

    Raises:
        AssertionError: If a and b are not the same type.
    """
    assert type(a) is type(b), "Both arrays must have the same type"
    size_a = dr.width(a)
    size_b = dr.width(b)
    c = dr.empty(type(a), size_a + size_b)
    dr.scatter(c, a, dr.arange(Int, size_a))
    dr.scatter(c, b, size_a + dr.arange(Int, size_b))
    return c


def compute_orthonomal_basis(n):
    sign = dr.copysign(1.0, n[2])
    a = -1.0 / (sign + n[2])
    b = n[0] * n[1] * a
    b1 = Array3(1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0])
    b2 = Array3(b, sign + n[1] * n[1] * a, -n[1])
    return b1, b2


def project_to_plane_3D(n, e):
    b1, b2 = compute_orthonomal_basis(n)
    r1 = dr.dot(e, dr.abs(b1))
    r2 = dr.dot(e, dr.abs(b2))
    return dr.sqrt(r1 * r1 + r2 * r2)


@dr.syntax
def line_sphere_intersection(a: Array3, b: Array3, c: Array3, r: Float):
    d = b - a
    L = a - c
    a = dr.dot(d, d)
    b = 2 * dr.dot(d, L)
    c = dr.dot(L, L) - r * r
    discriminant = b * b - 4 * a * c
    is_hit = discriminant > 0
    t0 = Float(dr.inf)
    t1 = Float(dr.inf)
    if is_hit:
        t0 = (-b - dr.sqrt(discriminant)) / (2 * a)
        t1 = (-b + dr.sqrt(discriminant)) / (2 * a)
    return is_hit, t0, t1
