from dataclasses import dataclass

from diff_wost.core.fwd import Array2i, Array3, Array4i, Bool, Float, Int, dr
from diff_wost.core.math import closest_point_line_segment
from diff_wost.render.interaction import ClosestSilhouettePointRecord3D


def is_silhouette_edge(x: Array3, a: Array3, b: Array3, n0: Array3, n1: Array3) -> Bool:
    d, pt, t = closest_point_line_segment(x, a, b)
    view_dir = pt - x
    dot0 = dr.dot(view_dir, n0)
    dot1 = dr.dot(view_dir, n1)
    return dot0 * dot1 < 0.0


@dr.syntax
def is_silhouette_edge_fcpw(
    x: Array3, a: Array3, b: Array3, n0: Array3, n1: Array3
) -> Bool:
    sign = -1.0
    precision = 1e-6
    _is_silhouette = Bool(False)

    d, pt, t = closest_point_line_segment(x, a, b)
    # edge is a silhouette if it concave and the query point lies on the edge
    if d <= precision:
        edge_dir = dr.normalize(b - a)
        signed_dihedral_angle = dr.atan2(
            dr.dot(edge_dir, dr.cross(n0, n1)), dr.dot(n0, n1)
        )
        _is_silhouette = sign * signed_dihedral_angle > precision
    else:
        view_dir = x - pt
        view_dir_unit = view_dir / d
        dot0 = dr.dot(view_dir_unit, n0)
        dot1 = dr.dot(view_dir_unit, n1)

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


@dataclass
class SilhouetteEdge:
    a: Array3  # first vertex of the edge
    b: Array3  # second vertex of the edge

    c: Array3  # first vertex of the triangle, vertex of abc
    d: Array3  # second vertex of the triangle, vertex of adb
    # indices: Array4i
    indices: Array4i  # vertex indices
    index: Int  # edge index
    face_indices: Array2i = Array2i(-1, -1)  # indices of the two faces
    prim_id: Int = Int(-1)  # prim_id in the new silhouette array

    def n0(self):
        return dr.normalize(dr.cross(self.b - self.a, self.c - self.a))

    def n1(self):
        return dr.normalize(dr.cross(self.d - self.a, self.b - self.a))

    @dr.syntax
    def star_radius(self, p: Array3, r_max: Float = Float(dr.inf)):
        d_min = Float(r_max)
        d, pt, t = closest_point_line_segment(p, self.a, self.b)
        if d < r_max:
            _is_silhouette = (self.indices[0] == -1) | (self.indices[3] == -1)
            if ~_is_silhouette:
                _is_silhouette = is_silhouette_edge(
                    p, self.a, self.b, self.n0(), self.n1()
                )
            if _is_silhouette:
                d_min = d
        return d_min

    @dr.syntax
    def closest_silhouette(self, x: Array3, r_max: Float = Float(dr.inf)):
        c_rec = dr.zeros(ClosestSilhouettePointRecord3D)
        d, pt, t = closest_point_line_segment(x, self.a, self.b)
        if d < r_max:
            _is_silhouette = (self.indices[0] == -1) | (self.indices[3] == -1)
            if ~_is_silhouette:
                _is_silhouette = is_silhouette_edge(
                    x, self.a, self.b, self.n0(), self.n1()
                )
            if _is_silhouette:
                c_rec = ClosestSilhouettePointRecord3D(
                    valid=Bool(True), p=pt, d=d, prim_id=self.prim_id
                )
        return c_rec
