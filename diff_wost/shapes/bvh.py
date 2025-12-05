from dataclasses import dataclass

import numpy as np

from diff_wost.core.fwd import PCG32, Array2, Array2i, Bool, Float, Int, UInt, dr
from diff_wost.render.interaction import (
    BoundarySamplingRecord,
    ClosestPointRecord,
    Intersection,
)
from diff_wost.shapes.line_segment import LineSegment
from diff_wost.shapes.point import Point2D
from diff_wost.shapes.silhouette_vertex import SilhouetteVertex


@dr.syntax
def project_to_plane(n, e):
    b = Array2(-n[1], n[0])
    r = dr.dot(e, dr.abs(b))
    return dr.abs(r)


@dataclass
class BoundingBox:
    p_min: Array2
    p_max: Array2

    def centroid(self):
        return (self.p_min + self.p_max) / 2

    def distance(self, p: Array2):
        u = self.p_min - p
        v = p - self.p_max
        d_min = dr.norm(dr.maximum(dr.maximum(u, v), 0.0))
        d_max = dr.norm(dr.minimum(u, v))
        return d_min, d_max

    def overlaps(self, c, r_max):
        d_min, d_max = self.distance(c)
        return d_min <= r_max, d_min, d_max

    @dr.syntax
    def intersect(self, x: Array2, v: Array2, r_max: Float = Float(dr.inf)):
        t0 = (self.p_min - x) / v
        t1 = (self.p_max - x) / v
        t_near = dr.minimum(t0, t1)
        t_far = dr.maximum(t0, t1)
        hit = Bool(False)
        t_near_max = dr.maximum(dr.max(t_near), 0.0)
        t_far_min = dr.minimum(dr.min(t_far), r_max)
        t_min = Float(dr.inf)
        t_max = Float(dr.inf)
        if t_near_max > t_far_min:
            hit = Bool(False)
        else:
            t_min = t_near_max
            t_max = t_far_min
            hit = Bool(True)
        return hit, t_min, t_max


@dataclass
class BVHNode:
    box: BoundingBox
    reference_offset: Int
    second_child_offset: Int
    n_references: Int

    @dr.syntax
    def is_leaf(self):
        return self.n_references > 0


@dataclass
class TraversalStack:
    index: Int
    distance: Float


class BVH:
    flat_tree: BVHNode
    primitives: LineSegment
    silhouettes: SilhouetteVertex

    def __init__(self, vertices: Array2, indices: Array2i):
        import diff_wost_ext as dw

        bvh = dw.BVH2LineSegment(vertices.numpy().T, indices.numpy().T)
        primitives = dw.convert_line_segments_2d(bvh.primitives())
        flat_tree: dw.BvhNodeSoA2 = dw.convert_bvh_nodes_2d(bvh.nodes())

        self.primitives = LineSegment(
            a=Array2(np.array(primitives.a).T),
            b=Array2(np.array(primitives.b).T),
            # index in the original array
            index=Int(np.array(primitives.index)),
            # index in the sorted array
            sorted_index=dr.arange(Int, len(primitives.index)),
        )

        self.flat_tree = BVHNode(
            box=BoundingBox(
                p_min=Array2(np.array(flat_tree.box.pMin).T),
                p_max=Array2(np.array(flat_tree.box.pMax).T),
            ),
            reference_offset=Int(np.array(flat_tree.referenceOffset)),
            second_child_offset=Int(np.array(flat_tree.referenceOffset)),
            n_references=Int(np.array(flat_tree.nReferences)),
        )

    @dr.syntax
    def closest_point(self, p: Array2):
        r_max = Float(dr.inf)
        its = ClosestPointRecord(
            valid=Bool(True),
            p=Array2(p),
            n=Array2(0.0, 0.0),
            t=Float(dr.inf),
            d=Float(dr.inf),
            prim_id=Int(-1),
        )
        stack_ptr = dr.zeros(Int, dr.width(p))
        stack = dr.alloc_local(TraversalStack, 64)
        stack[0] = TraversalStack(index=Int(0), distance=Float(r_max))
        while stack_ptr >= 0:
            # pop off the next node to work on
            stack_node = stack[UInt(stack_ptr)]
            node_index = stack_node.index
            current_distance = stack_node.distance
            stack_ptr -= 1
            if current_distance > r_max:
                pass
            else:
                node = dr.gather(BVHNode, self.flat_tree, node_index)
                if node.n_references > 0:
                    #     # leaf node
                    j = Int(0)
                    while j < node.n_references:
                        reference_index = node.reference_offset + j
                        prim = dr.gather(LineSegment, self.primitives, reference_index)
                        _its = prim.closest_point(p)
                        if _its.d < its.d:
                            its = _its
                        j += 1
                    r_max = dr.minimum(r_max, its.d)
                else:
                    # non-leaf node
                    node_left = dr.gather(BVHNode, self.flat_tree, node_index + 1)
                    node_right = dr.gather(
                        BVHNode, self.flat_tree, node_index + node.second_child_offset
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
                            if d_max_right < d_max_right:
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
    def intersect(self, x: Array2, v: Array2, r_max: Float = Float(dr.inf)):
        its = dr.zeros(Intersection)
        root_node = dr.gather(BVHNode, self.flat_tree, 0)
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
                    node = dr.gather(BVHNode, self.flat_tree, node_index)
                    if node.n_references > 0:
                        j = Int(0)
                        while j < node.n_references:
                            reference_index = node.reference_offset + j
                            prim = dr.gather(
                                LineSegment, self.primitives, reference_index
                            )
                            _its = prim.ray_intersect(x, v, r_max)
                            if _its.valid & (_its.d < r_max):
                                r_max = its.d
                                its = _its
                            j += 1
                    else:
                        left_box = dr.gather(
                            BoundingBox, self.flat_tree.box, node_index + 1
                        )
                        right_box = dr.gather(
                            BoundingBox,
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
    def branch_traversal_weight(self, x: Array2, R: Float, p: Array2):
        from diff_wost.render.greens_fn import GreensBall

        return dr.abs(GreensBall(c=x, R=R).G(x, p))
        # return Float(1.0)

    @dr.syntax
    def sample_boundary(self, x: Array2, R: Float, sampler: PCG32):
        b_rec = dr.zeros(BoundarySamplingRecord)
        hit = Bool(False)
        pdf = Float(1.0)
        sorted_prim_id = Int(-1)
        root_node = dr.gather(BVHNode, self.flat_tree, 0)
        overlap, d_min, d_max = root_node.box.overlaps(x, R)
        if overlap:
            hit, sorted_prim_id, pdf = self._sample_boundary(x, R, sampler)
        if hit:
            prim = dr.gather(LineSegment, self.primitives, sorted_prim_id)
            p, _pdf, t = prim.sample_point(sampler)
            b_rec = BoundarySamplingRecord(
                p=p,
                n=prim.normal(),
                t=t,
                pdf=pdf * _pdf,
                pmf=pdf,
                prim_id=prim.index,
                type=prim.type,
            )
        return hit, b_rec

    @dr.syntax
    def _sample_boundary(
        self, x: Array2, R: Float, sampler: PCG32, r_max: Float = Float(dr.inf)
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
            node = dr.gather(BVHNode, self.flat_tree, node_index)
            stack_ptr -= 1

            if node.is_leaf():
                # terminate at leaf node
                total_primitive_weight = Float(0.0)
                # traverse all primitives in the leaf node
                j = Int(0)
                while j < node.n_references:
                    reference_index = node.reference_offset + j
                    prim = dr.gather(LineSegment, self.primitives, reference_index)
                    surface_area = prim.surface_area()
                    if (r_max <= R) | prim.sphere_intersect(
                        x, R
                    ):  # all primitives are visible or hit
                        hit = Bool(True)
                        total_primitive_weight += surface_area
                        selection_prob = surface_area / total_primitive_weight
                        u = sampler.next_float32()
                        if u < selection_prob:  # select primitive
                            sorted_prim_id = prim.sorted_index
                    j += 1
                if total_primitive_weight > 0:
                    prim = dr.gather(LineSegment, self.primitives, sorted_prim_id)
                    surface_area = prim.surface_area()
                    pdf *= surface_area / total_primitive_weight
            else:
                box0 = dr.gather(BoundingBox, self.flat_tree.box, node_index + 1)
                overlap0, d_min0, d_max0 = box0.overlaps(x, R)
                weight0 = dr.select(overlap0, 1.0, 0.0)

                box1 = dr.gather(
                    BoundingBox,
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
                        node_index = node_index + 1  # jump to left child
                        pdf *= traversal_prob0
                        r_max = d_max0
                    else:
                        # choose right child
                        node_index = (
                            node_index + node.second_child_offset
                        )  # jump to right child
                        pdf *= traversal_prob1
                        r_max = d_max1

        return hit, sorted_prim_id, pdf


class PointBVH:
    flat_tree: BVHNode
    primitives: Point2D

    def __init__(self, vertices: Array2):
        import diff_wost_ext as dw

        c_bvh = dw.BVH2Point(vertices.numpy().T)
        primitives = dw.convert_points_2d(c_bvh.primitives())
        flat_tree = dw.convert_bvh_nodes_2d(c_bvh.nodes())

        self.primitives = Point2D(
            p=Array2(np.array(primitives.p).T),
            # index in the original array
            index=Int(np.array(primitives.index)),
            # index in the sorted array
            sorted_index=dr.arange(Int, len(primitives.index)),
        )

        self.flat_tree = BVHNode(
            box=BoundingBox(
                p_min=Array2(np.array(flat_tree.box.pMin).T),
                p_max=Array2(np.array(flat_tree.box.pMax).T),
            ),
            reference_offset=Int(np.array(flat_tree.referenceOffset)),
            second_child_offset=Int(np.array(flat_tree.referenceOffset)),
            n_references=Int(np.array(flat_tree.nReferences)),
        )

    @dr.syntax
    def closest_point(self, p: Array2):
        r_max = Float(dr.inf)
        its = ClosestPointRecord(
            valid=Bool(True),
            p=Array2(p),
            n=Array2(0.0, 0.0),
            t=Float(dr.inf),
            d=Float(dr.inf),
            prim_id=Int(-1),
        )
        stack_ptr = dr.zeros(Int, dr.width(p))
        stack = dr.alloc_local(TraversalStack, 64)
        stack[0] = TraversalStack(index=Int(0), distance=Float(r_max))
        while stack_ptr >= 0:
            # pop off the next node to work on
            stack_node = stack[UInt(stack_ptr)]
            node_index = stack_node.index
            current_distance = stack_node.distance
            stack_ptr -= 1
            if current_distance > r_max:
                pass
            else:
                node = dr.gather(BVHNode, self.flat_tree, node_index)
                if node.n_references > 0:
                    #     # leaf node
                    j = Int(0)
                    while j < node.n_references:
                        reference_index = node.reference_offset + j
                        prim = dr.gather(Point2D, self.primitives, reference_index)
                        _its = ClosestPointRecord(
                            valid=Bool(True),
                            p=prim.p,
                            n=Array2(0.0, 0.0),
                            t=Float(0.0),
                            d=dr.norm(prim.p - p),
                            prim_id=prim.index,
                        )
                        if _its.d < its.d:
                            its = _its
                        j += 1
                    r_max = dr.minimum(r_max, its.d)
                else:
                    # non-leaf node
                    node_left = dr.gather(BVHNode, self.flat_tree, node_index + 1)
                    node_right = dr.gather(
                        BVHNode, self.flat_tree, node_index + node.second_child_offset
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
                            if d_max_right < d_max_right:
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
    def branch_traversal_weight(self, x: Array2, R: Float, p: Array2):
        from diff_wost.render.greens_fn import GreensBall

        return dr.abs(GreensBall(c=x, R=R).G(x, p))
        return Float(1.0)

    @dr.syntax
    def sample_boundary(self, x: Array2, R: Float, sampler: PCG32):
        b_rec = dr.zeros(BoundarySamplingRecord)
        hit = Bool(False)
        pdf = Float(1.0)
        sorted_prim_id = Int(-1)
        root_node = dr.gather(BVHNode, self.flat_tree, 0)
        overlap, d_min, d_max = root_node.box.overlaps(x, R)
        if overlap:
            hit, sorted_prim_id, pdf = self._sample_boundary(x, R, sampler)
        if hit:
            prim = dr.gather(Point2D, self.primitives, sorted_prim_id)
            p, _pdf, t = prim.p, Float(1.0), Float(0.0)
            b_rec = BoundarySamplingRecord(
                p=p,
                n=Array2(0.0, 0.0),
                t=t,
                pdf=pdf * _pdf,
                pmf=pdf,
                prim_id=prim.index,
            )
        return hit, b_rec

    @dr.syntax
    def _sample_boundary(
        self, x: Array2, R: Float, sampler: PCG32, r_max: Float = Float(dr.inf)
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
            node = dr.gather(BVHNode, self.flat_tree, node_index)
            stack_ptr -= 1

            if node.is_leaf():
                # terminate at leaf node
                total_primitive_weight = Float(0.0)
                # traverse all primitives in the leaf node
                j = Int(0)
                while j < node.n_references:
                    reference_index = node.reference_offset + j
                    prim = dr.gather(Point2D, self.primitives, reference_index)
                    surface_area = Float(1.0)
                    if (r_max <= R) | (
                        dr.norm(prim.p - x) <= R
                    ):  # all primitives are visible or hit
                        hit = Bool(True)
                        total_primitive_weight += surface_area
                        selection_prob = surface_area / total_primitive_weight
                        u = sampler.next_float32()
                        if u < selection_prob:  # select primitive
                            sorted_prim_id = prim.sorted_index
                    j += 1
                if total_primitive_weight > 0:
                    prim = dr.gather(Point2D, self.primitives, sorted_prim_id)
                    surface_area = Float(1.0)
                    pdf *= surface_area / total_primitive_weight
            else:
                box0 = dr.gather(BoundingBox, self.flat_tree.box, node_index + 1)
                overlap0, d_min0, d_max0 = box0.overlaps(x, R)
                weight0 = dr.select(overlap0, 1.0, 0.0)

                box1 = dr.gather(
                    BoundingBox,
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
                        node_index = node_index + 1  # jump to left child
                        pdf *= traversal_prob0
                        r_max = d_max0
                    else:
                        # choose right child
                        node_index = (
                            node_index + node.second_child_offset
                        )  # jump to right child
                        pdf *= traversal_prob1
                        r_max = d_max1

        return hit, sorted_prim_id, pdf
