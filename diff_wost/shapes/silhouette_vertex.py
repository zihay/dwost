from dataclasses import dataclass

from diff_wost.core.fwd import Array2, Array3, Array3i, Bool, Float, Int, dr
from diff_wost.core.math import closest_point_line_segment, is_silhouette
from diff_wost.render.interaction import ClosestSilhouettePointRecord


@dataclass
class SilhouetteVertex:
    a: Array2
    b: Array2
    c: Array2
    indices: Array3i
    index: Int
    prim_id: Int = Int(-1)  # prim_id in the new silhouette array

    def n0(self):
        t = self.b - self.a
        n = Array2(t[1], -t[0])
        return dr.normalize(n)

    def n1(self):
        t = self.c - self.b
        n = Array2(t[1], -t[0])
        return dr.normalize(n)

    @dr.syntax
    def is_silhouette_vertex(self, x: Array2):
        return is_silhouette(x, self.a, self.b, self.c)
        sign = -1
        precision = 1e-4
        d = dr.norm(self.b - x)
        dir = dr.normalize(self.b - x)
        _is_silhouette = Bool(False)

        n0 = self.n0()
        n1 = self.n1()

        if d <= precision:
            # concave corner
            det = n0[0] * n1[1] - n0[1] * n1[0]
            _is_silhouette = sign * det > precision
        else:
            dot0 = dr.dot(dir, n0)
            dot1 = dr.dot(dir, n1)
            is_zero_dot0 = dr.abs(dot0) <= precision
            if is_zero_dot0:
                _is_silhouette = sign * dot1 > precision
            else:
                is_zero_dot1 = dr.abs(dot1) <= precision
                if is_zero_dot1:
                    _is_silhouette = sign * dot0 > precision
                else:
                    _is_silhouette = dot0 * dot1 < 0.0
        return _is_silhouette

    @dr.syntax
    def turning_angle(self):
        v1 = dr.normalize(self.b - self.a)
        v2 = dr.normalize(self.c - self.b)
        costheta = dr.dot(v1, v2)
        costheta = dr.clamp(costheta, -1.0, 1.0)
        return dr.acos(costheta)

    @dr.syntax
    def curvature(self):
        theta = self.turning_angle()
        w = dr.norm(self.a - self.c)
        return 2.0 * dr.sin(theta) / w

    @dr.syntax
    def star_radius(self, p: Array2, r_min: Float = dr.inf):
        d_min = r_min
        d = dr.norm(p - self.b)
        if d < r_min:
            if self.is_silhouette_vertex(p):
                d_min = d
        return d_min

    @dr.syntax
    def closest_silhouette(self, x: Array2, r_max: Float = Float(dr.inf)):
        c_rec = dr.zeros(ClosestSilhouettePointRecord)

        p = self.b
        view_dir = x - p
        d = dr.norm(view_dir)
        if (self.indices[0] != -1) & (self.indices[2] != -1):
            if d < r_max:
                if self.is_silhouette_vertex(x):
                    c_rec = ClosestSilhouettePointRecord(
                        valid=Bool(True), p=p, d=d, prim_id=self.prim_id
                    )
        return c_rec


def is_silhouette_edge(x: Array3, a: Array3, b: Array3, n0: Array3, n1: Array3) -> Bool:
    d, pt, t = closest_point_line_segment(x, a, b)
    view_dir = pt - x
    dot0 = dr.dot(view_dir, n0)
    dot1 = dr.dot(view_dir, n1)
    return dot0 * dot1 < 0.0
