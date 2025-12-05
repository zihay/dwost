from dataclasses import dataclass

import numpy as np

from diff_wost.core.fwd import (
    PCG32,
    Array2,
    Array3,
    Array3i,
    Array4i,
    Bool,
    Float,
    Int,
    UInt,
    dr,
)
from diff_wost.core.math import in_range
from diff_wost.render.interaction import (
    BoundarySamplingRecord3D,
    ClosestPointRecord3D,
    ClosestSilhouettePointRecord3D,
    Intersection3D,
)
from diff_wost.shapes.bvh import TraversalStack
from diff_wost.shapes.bvh3d import BoundingBox3D, BVHNode3D
from diff_wost.shapes.silhouette_edge import SilhouetteEdge
from diff_wost.shapes.triangle import Triangle


@dr.syntax
def compute_orthonomal_basis(n):
    sign = dr.copysign(1.0, n[2])
    a = -1.0 / (sign + n[2])
    b = n[0] * n[1] * a
    b1 = Array3(1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0])
    b2 = Array3(b, sign + n[1] * n[1] * a, -n[1])
    return b1, b2


@dr.syntax
def project_to_plane_3D(n, e):
    b1, b2 = compute_orthonomal_basis(n)
    r1 = dr.dot(e, dr.abs(b1))
    r2 = dr.dot(e, dr.abs(b2))
    return dr.sqrt(r1 * r1 + r2 * r2)


@dataclass
class BoundingCone3D:
    axis: Array3
    half_angle: Float
    radius: Float

    @dr.syntax
    def is_valid(self):
        return self.half_angle >= 0.0


@dataclass
class SNCHNode3D(BVHNode3D):
    cone: BoundingCone3D
    silhouette_reference_offset: int
    n_silhouette_references: int

    @dr.syntax
    def overlaps(self, x: Array3, r_max: Float = Float(dr.inf)):
        min_angle = Float(0.0)
        max_angle = Float(dr.pi / 2)
        is_overlap = Bool(False)
        d_min = Float(dr.inf)
        is_overlap, d_min, d_max = self.box.overlaps(x, r_max)
        if self.cone.is_valid() & is_overlap:
            # prune if the box is not hit
            is_overlap = (self.cone.half_angle > (dr.pi / 2)) & (
                d_min < dr.epsilon(Float)
            )
            if ~is_overlap:
                c = self.box.centroid()
                view_cone_axis = c - x
                l = dr.norm(view_cone_axis)
                view_cone_axis = view_cone_axis / l
                d_axis_angle = dr.acos(
                    dr.clamp(dr.dot(self.cone.axis, view_cone_axis), -1.0, 1.0)
                )
                is_overlap = in_range(
                    dr.pi / 2.0,
                    d_axis_angle - self.cone.half_angle,
                    d_axis_angle + self.cone.half_angle,
                )
                if ~is_overlap:
                    if l > self.cone.radius:
                        # NOTE: most of the pruning is done here
                        # outside the bounding sphere
                        view_cone_half_angle = dr.asin(self.cone.radius / l)
                        half_angle_sum = self.cone.half_angle + view_cone_half_angle
                        min_angle = d_axis_angle - half_angle_sum
                        max_angle = d_axis_angle + half_angle_sum
                        is_overlap = (half_angle_sum > (dr.pi / 2)) | in_range(
                            dr.pi / 2.0, min_angle, max_angle
                        )
                    else:
                        e = self.box.p_max - c
                        d = dr.dot(e, dr.abs(view_cone_axis))
                        s = l - d
                        is_overlap = s < 0.0
                        if ~is_overlap:
                            d = project_to_plane_3D(view_cone_axis, e)
                            view_cone_half_angle = dr.atan2(d, s)
                            half_angle_sum = self.cone.half_angle + view_cone_half_angle
                            min_angle = d_axis_angle - half_angle_sum
                            max_angle = d_axis_angle + half_angle_sum
                            is_overlap = (half_angle_sum > (dr.pi / 2)) | in_range(
                                dr.pi / 2.0, min_angle, max_angle
                            )
        return is_overlap, d_min

    @dr.syntax
    def overlaps_2(
        self,
        x: Array2,
        e: Array2,
        rho_bound: Float = Float(1.0),
        r_max: Float = Float(dr.inf),
        type: Int = Int(0),
    ):
        # used by star radius 2
        is_overlap = Bool(False)
        d_min = Float(dr.inf)
        is_overlap, d_min, d_max = self.box.overlaps(x, r_max)
        if self.cone.is_valid() & is_overlap:
            is_overlap = (self.cone.half_angle > (dr.pi / 2)) & (
                d_min < dr.epsilon(Float)
            )
            if ~is_overlap:
                c = self.box.centroid()
                view_cone_axis = c - x
                l = dr.norm(view_cone_axis)
                view_cone_axis = view_cone_axis / l

                is_overlap = l < self.cone.radius
                if ~is_overlap:
                    # angle between view_cone_axis and self.cone.axis
                    axis_angle = dr.acos(
                        dr.clamp(dr.dot(self.cone.axis, view_cone_axis), -1.0, 1.0)
                    )
                    view_cone_half_angle = dr.asin(self.cone.radius / l)
                    half_angle_sum = self.cone.half_angle + view_cone_half_angle
                    alpha_min = axis_angle - half_angle_sum
                    alpha_max = axis_angle + half_angle_sum

                    # probably has silhouettes
                    is_overlap = (half_angle_sum > (dr.pi / 2)) | in_range(
                        dr.pi / 2.0, alpha_min, alpha_max
                    )

                    if ~is_overlap:
                        # alpha_min > -pi/2 and alpha_max < pi/2
                        dn_min = dr.minimum(dr.cos(alpha_min), dr.cos(alpha_max))
                        dn_max = dr.maximum(dr.cos(alpha_min), dr.cos(alpha_max))

                        ed_axis_angle = dr.acos(
                            dr.clamp(dr.dot(e, view_cone_axis), -1.0, 1.0)
                        )
                        ed_angle_min = ed_axis_angle - self.cone.half_angle
                        ed_angle_max = ed_axis_angle + self.cone.half_angle
                        ed_min = dr.minimum(dr.cos(ed_angle_min), dr.cos(ed_angle_max))
                        ed_max = dr.maximum(dr.cos(ed_angle_min), dr.cos(ed_angle_max))
                        # Neumann boundary condition
                        en_axis_angle = dr.acos(
                            dr.clamp(dr.dot(e, self.cone.axis), -1.0, 1.0)
                        )
                        en_angle_min = en_axis_angle - self.cone.half_angle
                        en_angle_max = en_axis_angle + self.cone.half_angle
                        en_min = dr.minimum(dr.cos(en_angle_min), dr.cos(en_angle_max))
                        en_max = dr.maximum(dr.cos(en_angle_min), dr.cos(en_angle_max))

                        en_dn_max = en_max / dn_min
                        en_dn_min = Float(0.0)
                        if en_min < 0.0:
                            en_dn_min = en_min / dn_min
                        else:
                            en_dn_min = en_min / dn_max

                        def rho(ed, en_dn):
                            return 1.0 - 2.0 * ed * en_dn + dr.sqr(en_dn)

                        rho_max = rho(ed_min, en_dn_min)
                        rho_max = dr.maximum(rho_max, rho(ed_min, en_dn_max))
                        rho_max = dr.maximum(rho_max, rho(ed_max, en_dn_min))
                        rho_max = dr.maximum(rho_max, rho(ed_max, en_dn_max))

                        is_overlap = rho_max > rho_bound * rho_bound
        return is_overlap, d_min

    @dr.syntax
    def visit(
        self,
        x: Array3,
        v: Array3,
        rho_max: Float = Float(1.0),
        r_max: Float = Float(dr.inf),
    ):
        is_overlap = Bool(False)
        d_min = Float(dr.inf)
        # check box inside the query sphere
        is_overlap, d_min, d_max = self.box.overlaps(x, r_max)
        if self.cone.is_valid() & is_overlap:
            is_overlap = (self.cone.half_angle > (dr.pi / 2)) & (
                d_min < dr.epsilon(Float)
            )
            if ~is_overlap:
                c = self.box.centroid()
                view_cone_axis = c - x
                l = dr.norm(view_cone_axis)
                view_cone_axis = view_cone_axis / l
                is_overlap = l < self.cone.radius  # point inside the bounding sphere
                if ~is_overlap:
                    # angle between view_cone_axis and self.cone.axis
                    axis_angle = dr.acos(
                        dr.clamp(dr.dot(self.cone.axis, view_cone_axis), -1.0, 1.0)
                    )
                    view_cone_half_angle = dr.asin(self.cone.radius / l)
                    half_angle_sum = self.cone.half_angle + view_cone_half_angle
                    alpha_min = axis_angle - half_angle_sum
                    alpha_max = axis_angle + half_angle_sum

                    # probably has silhouettes
                    is_overlap = (half_angle_sum > (dr.pi / 2)) | in_range(
                        dr.pi / 2.0, alpha_min, alpha_max
                    )

                    if ~is_overlap:
                        # alpha_min > -pi/2 and alpha_max < pi/2
                        cos_min_alpha = dr.minimum(dr.cos(alpha_min), dr.cos(alpha_max))
                        cos_max_alpha = dr.maximum(dr.cos(alpha_min), dr.cos(alpha_max))
                        if (alpha_min <= 0.0) & (alpha_max >= 0.0):
                            cos_max_alpha = dr.maximum(cos_max_alpha, 1.0)

                        beta = dr.acos(dr.clamp(dr.dot(self.cone.axis, v), -1.0, 1.0))
                        beta_min = beta - self.cone.half_angle
                        beta_max = beta + self.cone.half_angle
                        cos_min_beta = dr.minimum(dr.cos(beta_min), dr.cos(beta_max))
                        cos_max_beta = dr.maximum(dr.cos(beta_min), dr.cos(beta_max))
                        if (beta_min <= 0.0) & (beta_max >= 0.0):
                            cos_max_beta = dr.maximum(cos_max_beta, 1.0)

                        gamma = dr.acos(dr.clamp(dr.dot(v, view_cone_axis), -1.0, 1.0))
                        gamma_min = gamma - view_cone_half_angle
                        gamma_max = gamma + view_cone_half_angle
                        cos_min_gamma = dr.minimum(dr.cos(gamma_min), dr.cos(gamma_max))
                        cos_max_gamma = dr.maximum(dr.cos(gamma_min), dr.cos(gamma_max))
                        if (gamma_min <= 0.0) & (gamma_max >= 0.0):
                            cos_max_gamma = dr.maximum(cos_max_gamma, 1.0)

                        a_min = cos_min_beta / cos_max_alpha
                        a_max = cos_max_beta / cos_min_alpha

                        b_min = cos_min_gamma
                        b_max = cos_max_gamma

                        rho2max = (
                            1
                            - 2.0
                            * dr.minimum(
                                dr.minimum(a_min * b_min, a_min * b_max),
                                dr.minimum(a_max * b_min, a_max * b_max),
                            )
                            + dr.maximum(a_min * a_min, a_max * a_max)
                        )
                        is_overlap = rho2max > rho_max * rho_max

        return is_overlap, d_min


@dataclass
class SNCH3D:
    flat_tree: SNCHNode3D
    primitives: Triangle
    silhouettes: SilhouetteEdge
    node_visited: Int

    def __init__(self, vertices: Array3, indices: Array3i):
        import diff_wost_ext as dw

        snch = dw.Snch3(vertices.numpy().T, indices.numpy().T)
        snch_nodes = dw.convert_snch_nodes_3d(snch.nodes())
        snch_prims = dw.convert_triangles(snch.primitives())
        # Use silhouette_refs_soa (reordered for SNCH traversal, silhouette_reference_offset points into this)
        snch_silhouettes = snch.silhouette_refs_soa()

        # Use BVH-sorted primitives from C++ (reference_offset in nodes points into this array)
        self.primitives = Triangle(
            a=Array3(np.array(snch_prims.a).T),
            b=Array3(np.array(snch_prims.b).T),
            c=Array3(np.array(snch_prims.c).T),
            # index in the original array
            index=Int(np.array(snch_prims.index)),
            # index in the sorted array
            sorted_index=dr.arange(Int, len(snch_prims.index)),
        )
        # Use reordered silhouettes (silhouette_reference_offset in nodes points into this array)
        self.silhouettes = SilhouetteEdge(
            a=Array3(np.array(snch_silhouettes.a).T),
            b=Array3(np.array(snch_silhouettes.b).T),
            c=Array3(np.array(snch_silhouettes.c).T),
            d=Array3(np.array(snch_silhouettes.d).T),
            index=Int(np.array(snch_silhouettes.p_index)),
            indices=Array4i(np.array(snch_silhouettes.indices)),
            prim_id=dr.arange(Int, len(snch_silhouettes.p_index)),
        )

        box = BoundingBox3D(
            p_min=Array3(np.array(snch_nodes.box.pMin).T),
            p_max=Array3(np.array(snch_nodes.box.pMax).T),
        )
        cone = BoundingCone3D(
            axis=Array3(np.array(snch_nodes.cone.axis).T),
            half_angle=Float(np.array(snch_nodes.cone.halfAngle)),
            radius=Float(np.array(snch_nodes.cone.radius)),
        )
        self.flat_tree = SNCHNode3D(
            box=box,
            reference_offset=Int(np.array(snch_nodes.reference_offset)),
            second_child_offset=Int(np.array(snch_nodes.reference_offset)),
            n_references=Int(np.array(snch_nodes.n_references)),
            silhouette_reference_offset=Int(
                np.array(snch_nodes.silhouette_reference_offset)
            ),
            n_silhouette_references=Int(np.array(snch_nodes.n_silhouette_references)),
            cone=cone,
        )
        self.node_visited = Int(0)

    @dr.syntax
    def closest_point(self, p: Array3):
        p = Array3(p)
        r_max = Float(dr.inf)
        its = ClosestPointRecord3D(
            p=Array3(0, 0, 0),
            n=Array3(0, 0, 0),
            uv=Array2(0, 0),
            d=Float(dr.inf),
            prim_id=Int(-1),
        )
        stack_ptr = dr.zeros(Int, dr.width(p))
        stack = dr.alloc_local(TraversalStack, 64)
        stack[0] = TraversalStack(index=Int(0), distance=Float(r_max))
        while stack_ptr >= 0:
            stack_node = stack[UInt(stack_ptr)]
            node_index = stack_node.index
            current_distance = stack_node.distance
            stack_ptr -= 1
            if current_distance <= r_max:
                node = dr.gather(SNCHNode3D, self.flat_tree, node_index)
                if node.is_leaf():
                    # leaf node
                    j = Int(0)
                    while j < node.n_references:
                        reference_index = node.reference_offset + j
                        prim = dr.gather(Triangle, self.primitives, reference_index)
                        _its = prim.closest_point(p)
                        if _its.d < its.d:
                            its = _its
                            r_max = dr.minimum(r_max, its.d)
                        j += 1
                else:
                    node_left = dr.gather(SNCHNode3D, self.flat_tree, node_index + 1)
                    node_right = dr.gather(
                        SNCHNode3D,
                        self.flat_tree,
                        node_index + node.second_child_offset,
                    )
                    hit_left, d_min_left, d_max_left = node_left.box.overlaps(p, r_max)
                    r_max = dr.minimum(r_max, d_max_left)
                    hit_right, d_min_right, d_max_right = node_right.box.overlaps(
                        p, r_max
                    )
                    r_max = dr.minimum(r_max, d_max_right)

                    if hit_left & hit_right:
                        closer = node_index + 1
                        other = node_index + node.second_child_offset
                        if (d_min_left == 0.0) & (d_min_right == 0.0):
                            if d_max_right < d_max_left:
                                closer, other = other, closer
                        if d_min_right < d_min_left:
                            closer, other = other, closer
                            d_min_left, d_min_right = d_min_right, d_min_left

                        stack_ptr += 1
                        stack[UInt(stack_ptr)] = TraversalStack(
                            index=other, distance=d_min_right
                        )
                        stack_ptr += 1
                        stack[UInt(stack_ptr)] = TraversalStack(
                            index=closer, distance=d_min_left
                        )
                    elif hit_left:
                        stack_ptr += 1
                        stack[UInt(stack_ptr)] = TraversalStack(
                            index=node_index + 1, distance=d_min_left
                        )
                    elif hit_right:
                        stack_ptr += 1
                        stack[UInt(stack_ptr)] = TraversalStack(
                            index=node_index + node.second_child_offset,
                            distance=d_min_right,
                        )
        return its

    @dr.syntax
    def intersect(self, x: Array3, v: Array3, r_max: Float = Float(dr.inf)):
        its = dr.zeros(Intersection3D)
        root_node = dr.gather(SNCHNode3D, self.flat_tree, 0)
        hit, t_min, t_max = root_node.box.intersect(x, v, r_max)
        if hit:
            stack = dr.alloc_local(TraversalStack, 64)
            stack_ptr = dr.zeros(Int, dr.width(x))
            stack[0] = TraversalStack(index=Int(0), distance=Float(t_min))
            while stack_ptr >= 0:
                stack_node = stack[UInt(stack_ptr)]
                node_index = stack_node.index
                curr_dist = stack_node.distance
                stack_ptr -= 1
                if curr_dist <= r_max:
                    # prune curr_dist > r_max
                    node = dr.gather(SNCHNode3D, self.flat_tree, node_index)
                    if node.is_leaf():
                        j = Int(0)
                        while j < node.n_references:
                            reference_index = node.reference_offset + j
                            prim = dr.gather(Triangle, self.primitives, reference_index)
                            _its = prim.ray_intersect(x, v, r_max)
                            if _its.valid & (_its.d < r_max):
                                r_max = _its.d
                                its = _its
                            j += 1
                    else:
                        left_box = dr.gather(
                            BoundingBox3D, self.flat_tree.box, node_index + 1
                        )
                        right_box = dr.gather(
                            BoundingBox3D,
                            self.flat_tree.box,
                            node_index + node.second_child_offset,
                        )
                        hit0, t_min0, t_max0 = left_box.intersect(x, v, r_max)
                        hit1, t_min1, t_max1 = right_box.intersect(x, v, r_max)
                        if hit0 & hit1:
                            closer = node_index + 1
                            other = node_index + node.second_child_offset
                            if t_min1 < t_min0:  # swap to make sure left is closer
                                closer, other = other, closer
                                t_min0, t_min1 = t_min1, t_min0
                                t_max0, t_max1 = t_max1, t_max0

                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=other, distance=t_min1
                            )
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=closer, distance=t_min0
                            )
                        elif hit0:
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=node_index + 1, distance=t_min0
                            )
                        elif hit1:
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=node_index + node.second_child_offset,
                                distance=t_min1,
                            )
        return its

    @dr.syntax
    def branch_traversal_weight(self, x: Array3, R: Float, p: Array3):
        return Float(1.0)

    @dr.syntax
    def sample_boundary(
        self, x: Array3, R: Float, sampler: PCG32
    ) -> tuple[Bool, BoundarySamplingRecord3D]:
        b_rec = dr.zeros(BoundarySamplingRecord3D)
        hit = Bool(False)
        pdf = Float(1.0)
        sorted_prim_id = Int(-1)
        root_node = dr.gather(SNCHNode3D, self.flat_tree, 0)
        overlap, d_min, d_max = root_node.box.overlaps(x, R)
        if overlap:
            hit, sorted_prim_id, pdf = self._sample_boundary(x, R, sampler)
        if hit:
            prim = dr.gather(Triangle, self.primitives, sorted_prim_id)
            s_rec = prim.sample_point(sampler)
            b_rec = BoundarySamplingRecord3D(
                p=s_rec.p,
                n=s_rec.n,
                uv=s_rec.uv,
                pdf=pdf * s_rec.pdf,
                pmf=pdf,
                prim_id=prim.index,
                type=prim.type,
            )
        return hit, b_rec

    @dr.syntax
    def _sample_boundary(
        self, x: Array3, R: Float, sampler: PCG32, r_max: Float = Float(dr.inf)
    ):
        # a single root to leaf traversal
        d_max0 = Float(dr.inf)
        d_max1 = Float(dr.inf)

        hit = Bool(False)
        pdf = Float(1.0)
        sorted_prim_id = Int(-1)

        stack_ptr = dr.zeros(Int, dr.width(x))
        node_index = Int(0)
        while stack_ptr >= 0:
            node = dr.gather(SNCHNode3D, self.flat_tree, node_index)
            stack_ptr -= 1

            if node.is_leaf():
                # terminate at leaf node
                total_primitive_weight = Float(0.0)
                # traverse all primitives in the leaf node
                j = Int(0)
                while j < node.n_references:
                    reference_index = node.reference_offset + j
                    prim = dr.gather(Triangle, self.primitives, reference_index)
                    surface_area = prim.surface_area()
                    if ((r_max <= R) | prim.sphere_intersect(x, R)) & (
                        surface_area > 0.0
                    ):
                        hit = Bool(True)
                        total_primitive_weight += surface_area
                        selection_prob = surface_area / total_primitive_weight
                        u = sampler.next_float32()
                        if u < selection_prob:  # select primitive
                            u = u / selection_prob  # rescale u to [0, 1]
                            sorted_prim_id = prim.sorted_index
                    j += 1
                if total_primitive_weight > 0:
                    prim = dr.gather(Triangle, self.primitives, sorted_prim_id)
                    surface_area = prim.surface_area()
                    pdf *= surface_area / total_primitive_weight
            else:
                box0 = dr.gather(BoundingBox3D, self.flat_tree.box, node_index + 1)
                overlap0, d_min0, d_max0 = box0.overlaps(x, R)
                weight0 = dr.select(overlap0, 1.0, 0.0)

                box1 = dr.gather(
                    BoundingBox3D,
                    self.flat_tree.box,
                    node_index + node.second_child_offset,
                )
                overlap1, d_min1, d_max1 = box1.overlaps(x, R)
                weight1 = dr.select(overlap1, 1.0, 0.0)

                if weight0 > 0:
                    weight0 *= self.branch_traversal_weight(x, R, box0.centroid())

                if weight1 > 0:
                    weight1 *= self.branch_traversal_weight(x, R, box1.centroid())

                total_weight = weight0 + weight1
                if total_weight > 0:
                    stack_ptr += 1  # push a node
                    traversal_prob0 = weight0 / total_weight
                    traversal_prob1 = 1.0 - traversal_prob0
                    u = sampler.next_float32()
                    if u < traversal_prob0:
                        # choose left child
                        u = u / traversal_prob0  # rescale u to [0, 1]
                        node_index = node_index + 1  # jump to left child
                        pdf *= traversal_prob0
                        r_max = d_max0
                    else:
                        # choose right child
                        u = (
                            u - traversal_prob0
                        ) / traversal_prob1  # rescale u to [0, 1]
                        node_index = (
                            node_index + node.second_child_offset
                        )  # jump to right child
                        pdf *= traversal_prob1
                        r_max = d_max1

        return hit, sorted_prim_id, pdf

    @dr.syntax
    def closest_silhouette(self, x: Array3, r_max: Float = Float(dr.inf)):
        self.node_visited = dr.zeros(Int, dr.width(x))
        r_max = Float(r_max)
        c_rec = dr.zeros(ClosestSilhouettePointRecord3D)
        root_node = dr.gather(SNCHNode3D, self.flat_tree, 0)
        overlap, d_min, d_max = root_node.box.overlaps(x, r_max)
        if overlap:
            stack = dr.alloc_local(TraversalStack, 64)
            stack_ptr = dr.zeros(Int, dr.width(x))
            stack[0] = TraversalStack(index=Int(0), distance=d_min)
            while stack_ptr >= 0:
                stack_node = stack[UInt(stack_ptr)]
                node_index = stack_node.index
                curr_dist = stack_node.distance
                stack_ptr -= 1
                if curr_dist <= r_max:
                    # prune curr_dist > r_max
                    node = dr.gather(SNCHNode3D, self.flat_tree, node_index)
                    if node.is_leaf():  # leaf node
                        j = Int(0)
                        while j < node.n_silhouette_references:
                            reference_index = node.silhouette_reference_offset + j
                            silhouette = dr.gather(
                                SilhouetteEdge, self.silhouettes, reference_index
                            )
                            _c_rec = silhouette.closest_silhouette(x, r_max)
                            if _c_rec.valid & (_c_rec.d < r_max):
                                r_max = dr.minimum(r_max, _c_rec.d)
                                c_rec = _c_rec
                            j += 1
                        self.node_visited += 1
                    else:  # non-leaf node
                        left = dr.gather(SNCHNode3D, self.flat_tree, node_index + 1)
                        right = dr.gather(
                            SNCHNode3D,
                            self.flat_tree,
                            node_index + node.second_child_offset,
                        )

                        hit0, d_min0 = left.overlaps(x, r_max)
                        hit1, d_min1 = right.overlaps(x, r_max)

                        if hit0 & hit1:
                            closer = node_index + 1
                            other = node_index + node.second_child_offset
                            if d_min1 < d_min0:
                                closer, other = other, closer
                                d_min0, d_min1 = d_min1, d_min0
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=other, distance=d_min1
                            )
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=closer, distance=d_min0
                            )
                        elif hit0:
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=node_index + 1, distance=d_min0
                            )
                        elif hit1:
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=node_index + node.second_child_offset,
                                distance=d_min1,
                            )
                        self.node_visited += 1
        return c_rec

    @dr.syntax
    def star_radius(self, x: Array3, r_max: Float = Float(dr.inf)):
        c_rec = self.closest_silhouette(x, r_max)
        return c_rec.d

    @dr.syntax
    def star_radius_2(
        self,
        x: Array3,
        e: Array3,
        rho_max: Float = Float(1.0),
        r_max: Float = Float(dr.inf),
        type: Int = Int(0),
    ):
        r_max = Float(r_max)
        root_node = dr.gather(SNCHNode3D, self.flat_tree, 0)
        overlap, d_min, d_max = root_node.box.overlaps(x, r_max)
        if overlap:
            stack = dr.alloc_local(TraversalStack, 64)
            stack_ptr = dr.zeros(Int, dr.width(x))
            stack[0] = TraversalStack(index=Int(0), distance=d_min)
            while stack_ptr >= 0:
                stack_node = stack[UInt(stack_ptr)]
                node_index = stack_node.index
                curr_dist = stack_node.distance
                stack_ptr -= 1
                if curr_dist <= r_max:
                    # prune curr_dist > r_max
                    node = dr.gather(SNCHNode3D, self.flat_tree, node_index)
                    if node.is_leaf():  # leaf node
                        j = Int(0)
                        while j < node.n_references:
                            reference_index = node.reference_offset + j
                            prim = dr.gather(Triangle, self.primitives, reference_index)
                            _r_max = prim.star_radius_2(x, e, rho_max, r_max)
                            if _r_max < r_max:
                                r_max = _r_max
                            j += 1
                    else:  # non-leaf node
                        left = dr.gather(SNCHNode3D, self.flat_tree, node_index + 1)
                        right = dr.gather(
                            SNCHNode3D,
                            self.flat_tree,
                            node_index + node.second_child_offset,
                        )

                        hit0, d_min0 = left.overlaps_2(x, e, rho_max, r_max, type)
                        hit1, d_min1 = right.overlaps_2(x, e, rho_max, r_max, type)

                        if hit0 & hit1:
                            closer = node_index + 1
                            other = node_index + node.second_child_offset
                            if d_min1 < d_min0:
                                closer, other = other, closer
                                d_min0, d_min1 = d_min1, d_min0
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=other, distance=d_min1
                            )
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=closer, distance=d_min0
                            )
                        elif hit0:
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=node_index + 1, distance=d_min0
                            )
                        elif hit1:
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=node_index + node.second_child_offset,
                                distance=d_min1,
                            )
        return r_max

    @dr.syntax
    def star_radius_3(
        self,
        x: Array3,
        v: Array3,
        rho_max: Float = Float(1.0),
        r_max: Float = Float(dr.inf),
    ):
        r_max = Float(r_max)
        root_node = dr.gather(SNCHNode3D, self.flat_tree, 0)
        overlap, d_min, d_max = root_node.box.overlaps(x, r_max)
        if overlap:
            stack = dr.alloc_local(TraversalStack, 64)
            stack_ptr = dr.zeros(Int, dr.width(x))
            stack[0] = TraversalStack(index=Int(0), distance=d_min)
            while stack_ptr >= 0:
                stack_node = stack[UInt(stack_ptr)]
                node_index = stack_node.index
                curr_dist = stack_node.distance
                stack_ptr -= 1
                if curr_dist <= r_max:
                    # prune curr_dist > r_max
                    node = dr.gather(SNCHNode3D, self.flat_tree, node_index)
                    if node.is_leaf():  # leaf node
                        j = Int(0)
                        while j < node.n_references:
                            reference_index = node.reference_offset + j
                            prim = dr.gather(Triangle, self.primitives, reference_index)
                            _r_max = prim.star_radius_2(x, v, rho_max, r_max)
                            if _r_max < r_max:
                                r_max = _r_max
                            j += 1
                    else:  # non-leaf node
                        left = dr.gather(SNCHNode3D, self.flat_tree, node_index + 1)
                        right = dr.gather(
                            SNCHNode3D,
                            self.flat_tree,
                            node_index + node.second_child_offset,
                        )

                        hit0, d_min0 = left.visit(x, v, rho_max, r_max)

                        hit1, d_min1 = right.visit(x, v, rho_max, r_max)

                        if hit0 & hit1:
                            closer = node_index + 1
                            other = node_index + node.second_child_offset
                            if d_min1 < d_min0:
                                closer, other = other, closer
                                d_min0, d_min1 = d_min1, d_min0
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=other, distance=d_min1
                            )
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=closer, distance=d_min0
                            )
                        elif hit0:
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=node_index + 1, distance=d_min0
                            )
                        elif hit1:
                            stack_ptr += 1
                            stack[UInt(stack_ptr)] = TraversalStack(
                                index=node_index + node.second_child_offset,
                                distance=d_min1,
                            )

        return r_max
