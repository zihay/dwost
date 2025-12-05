from dataclasses import dataclass

from diff_wost.core.fwd import PCG32, Array2, Array3, Bool, Float, Int, dr
from diff_wost.core.math import (
    closest_point_triangle,
    inside_triangle,
    line_sphere_intersection,
)
from diff_wost.render.interaction import (
    BoundarySamplingRecord3D,
    ClosestPointRecord3D,
    Intersection3D,
)
from diff_wost.shapes.plane import Plane


@dataclass
class Triangle:
    a: Array3
    b: Array3
    c: Array3
    index: Int
    sorted_index: Int
    type: Int = Int(-1)

    @dr.syntax
    def centroid(self):
        return (self.a + self.b + self.c) / 3.0

    @dr.syntax
    def surface_area(self):
        return dr.norm(dr.cross(self.b - self.a, self.c - self.a)) / 2.0

    @dr.syntax
    def normal(self):
        return dr.normalize(dr.cross(self.b - self.a, self.c - self.a))

    @dr.syntax
    def sample_point(self, sampler: PCG32):
        u1 = dr.sqrt(sampler.next_float32())
        u2 = sampler.next_float32()
        u = 1.0 - u1
        v = u2 * u1
        w = 1.0 - u - v
        uv = Array2(u, v)
        p = self.a * u + self.b * v + self.c * w
        n = dr.cross(self.b - self.a, self.c - self.a)
        pdf = 2.0 / dr.norm(n)
        n = dr.normalize(n)
        return BoundarySamplingRecord3D(
            p=p, n=n, uv=uv, pdf=pdf, prim_id=self.sorted_index, type=self.type
        )

    @dr.syntax
    def sphere_intersect(self, x: Array3, R: Float):
        pt, uv, d = closest_point_triangle(x, self.a, self.b, self.c)
        return d <= R

    @dr.syntax
    def ray_intersect(self, x: Array2, d: Array2, r_max: Float):
        its = dr.zeros(Intersection3D)
        v1 = self.b - self.a
        v2 = self.c - self.a
        p = dr.cross(d, v2)
        det = dr.dot(v1, p)
        if dr.abs(det) > dr.epsilon(Float):
            inv_det = 1.0 / det
            s = x - self.a
            v = dr.dot(s, p) * inv_det
            if (v >= 0) & (v <= 1):
                q = dr.cross(s, v1)
                w = dr.dot(d, q) * inv_det
                if (w >= 0) & (v + w <= 1):
                    t = dr.dot(v2, q) * inv_det
                    if (t >= 0) & (t <= r_max):
                        its = Intersection3D(
                            valid=Bool(True),
                            p=self.a + v1 * v + v2 * w,
                            n=dr.normalize(dr.cross(v1, v2)),
                            uv=Array2(1.0 - v - w, v),
                            d=t,
                            prim_id=self.sorted_index,
                            on_boundary=Bool(True),
                            type=self.type,
                        )
        return its

    @dr.syntax
    def closest_point(self, p: Array2):
        pt, uv, d = closest_point_triangle(p, self.a, self.b, self.c)
        return ClosestPointRecord3D(
            valid=Bool(True),
            p=pt,
            n=self.normal(),
            uv=uv,
            d=d,
            prim_id=self.index,
            type=self.type,
        )

    @dr.syntax
    def is_inside_circle(self, c: Array3, r: Float):
        return (
            (dr.norm(self.a - c) < r)
            & (dr.norm(self.b - c) < r)
            & (dr.norm(self.c - c) < r)
        )

    @dr.syntax
    def star_radius_2(
        self, x: Array3, e: Array3, rho_max: Float, r_max: Float = Float(dr.inf)
    ):
        """Compute the star-shaped radius for gradient estimation.

        Args:
            x: Query point.
            e: Direction vector for gradient estimation.
            rho_max: Maximum rho parameter.
            r_max: Maximum radius to consider.

        Returns:
            Star-shaped radius value.
        """
        R = Float(r_max)
        c, r, p = Plane(self.a, self.normal()).bounding_point(x, e, rho_max)

        q = self.closest_point(x).p
        if self.is_inside_circle(c, r):
            # case 0: triangle is inside the circle
            pass
        elif dr.norm(q - c) > r:
            # case 1: triangle is outside the circle
            R = dr.minimum(R, dr.norm(q - x))
        elif inside_triangle(p, self.a, self.b, self.c):
            # case 2: critical point is inside the triangle
            R = dr.minimum(R, dr.norm(p - x))
        else:
            # case 3: check the edges of the triangle
            vertices = [self.a, self.b, self.c]
            for i in range(3):
                a = vertices[i]
                b = vertices[(i + 1) % 3]
                is_hit, t0, t1 = line_sphere_intersection(a, b, c, r)
                if is_hit:
                    if (t0 >= 0) & (t0 <= 1):
                        edge_p = a + t0 * (b - a)
                        R = dr.minimum(R, dr.norm(edge_p - x))
                    if (t1 >= 0) & (t1 <= 1):
                        edge_p = a + t1 * (b - a)
                        R = dr.minimum(R, dr.norm(edge_p - x))
        return R
