import numpy as np

from diff_wost.core.distr import DiscreteDistribution
from diff_wost.core.fwd import (
    PCG32,
    Array2,
    Array2i,
    Array3,
    Array3i,
    Bool,
    Float,
    Int,
    RayEpsilon,
    dr,
)
from diff_wost.core.math import distance_to_plane, ray_intersection
from diff_wost.render.interaction import (
    BoundarySamplingRecord,
    ClosestPointRecord,
    ClosestSilhouettePointRecord,
    Intersection,
    SilhouetteSamplingRecord,
)
from diff_wost.shapes.bvh import BVH, BoundingBox, PointBVH
from diff_wost.shapes.primitive import BoundaryType
from diff_wost.shapes.silhouette_vertex import SilhouetteVertex
from diff_wost.shapes.snch import SNCH


class Polyline:
    vertices: Array2
    indices: Array2i

    silhouettes: SilhouetteVertex = None

    bvh: BVH = None
    snch: SNCH = None
    silhouette_bvh: PointBVH = None

    normals: Array2 = None
    edge_normals: Array2 = None

    v2e: Array2i = None  # vertex to neighbor edges

    edge_pmf: DiscreteDistribution = None  # for boundary sampling
    silhouette_pmf: DiscreteDistribution = None  # for silhouette sampling

    use_bvh: bool = True

    length: Float = None

    def g(self, c_rec: ClosestPointRecord) -> Float:
        # Dirichlet boundary condition
        return self.u(c_rec.p)

    def h(self, b_rec: BoundarySamplingRecord) -> Float:
        # Neumann boundary condition
        return dr.dot(b_rec.n, self.dudx(b_rec.p))

    def __init__(self, vertices: Array2, indices: Array2i):
        self.vertices = vertices
        self.indices = indices
        self.configure()

    def configure(self):
        # compute bbox
        self.bbox = BoundingBox(
            p_min=Array2(dr.min(self.vertices.x), dr.min(self.vertices.y)),
            p_max=Array2(dr.max(self.vertices.x), dr.max(self.vertices.y)),
        )

        self.configure_normals()
        self.configure_edge_pmf()
        self.configure_silhouette()
        self.configure_v2e()
        self.configure_length()
        self.bvh = BVH(self.vertices, self.indices)
        self.snch = SNCH(self.vertices, self.indices)

    def configure_length(self):
        a = dr.gather(Array2, self.vertices, self.indices.x)
        b = dr.gather(Array2, self.vertices, self.indices.y)
        self.length = dr.sum(dr.norm(b - a))

    def configure_v2e(self):
        indices = self.indices.numpy().T
        v2e = {}
        for i in range(len(indices)):
            f = indices[i]
            a, b = f
            if a not in v2e:
                v2e[a] = [-1, -1]
            if b not in v2e:
                v2e[b] = [-1, -1]
            v2e[a][1] = i
            v2e[b][0] = i
        _v2e = []
        for i in range(dr.width(self.vertices)):
            _v2e.append(v2e[i])
        self.v2e = Array2i(np.array(_v2e, dtype=np.int32).T)

    def configure_silhouette(self):
        """
        Vectorized construction of silhouette vertices.

        A silhouette vertex connects exactly two directed edges: prev->v and v->next.
        We derive prev and next for all vertices using NumPy indexing, and then
        build the packed silhouette structure in one pass.
        """
        # Convert to NumPy once
        vertices_np = self.vertices.numpy()  # shape: (2, V)
        edges_np = self.indices.numpy()  # shape: (2, E)

        V = vertices_np.shape[1]
        if edges_np.size == 0:
            return

        # prev[v] = a for each edge (a -> b)
        # next[v] = b for each edge (a -> b)
        prev_idx = np.full(V, -1, dtype=np.int32)
        next_idx = np.full(V, -1, dtype=np.int32)
        a_idx = edges_np[0]
        b_idx = edges_np[1]
        prev_idx[b_idx] = a_idx
        next_idx[a_idx] = b_idx

        # A vertex is a silhouette if it has both prev and next
        is_silhouette = (prev_idx != -1) & (next_idx != -1)
        if not np.any(is_silhouette):
            return

        v_idx = np.nonzero(is_silhouette)[0].astype(np.int32)
        a_sel = prev_idx[v_idx]
        b_sel = v_idx
        c_sel = next_idx[v_idx]

        # Gather positions
        a_pts = vertices_np[:, a_sel]  # (2, N)
        b_pts = vertices_np[:, b_sel]  # (2, N)
        c_pts = vertices_np[:, c_sel]  # (2, N)

        # Pack into a temporary structure
        silhouettes = SilhouetteVertex(
            a=Array2(a_pts),
            b=Array2(b_pts),
            c=Array2(c_pts),
            indices=Array3i(np.vstack([a_sel, b_sel, c_sel])),
            index=Int(np.arange(v_idx.shape[0], dtype=np.int32)),
        )

        # Filter near-collinear vertices (keep meaningful corners)
        t1 = dr.normalize(silhouettes.b - silhouettes.a)  # prev->curr
        t2 = dr.normalize(silhouettes.c - silhouettes.b)  # curr->next
        collinearity = dr.abs(dr.dot(t1, t2))
        self.silhouettes = dr.gather(
            SilhouetteVertex, silhouettes, dr.compress(collinearity < (1.0 - 1e-5))
        )

        # Uniform sampling over silhouette vertices
        self.silhouette_pmf = DiscreteDistribution(
            dr.ones(Float, dr.width(self.silhouettes))
        )
        if self.use_bvh:
            self.silhouette_bvh = PointBVH(self.silhouettes.b)

    def configure_edge_pmf(self):
        p0 = dr.gather(Array2, self.vertices, self.indices.x)
        p1 = dr.gather(Array2, self.vertices, self.indices.y)
        # print("avg edge length", dr.mean(l))
        # print("min edge length", dr.min(l))
        self.edge_pmf = DiscreteDistribution(dr.norm(p1 - p0))

    def configure_normals(self):
        """Compute and store edge and vertex normals."""
        a = dr.gather(Array2, self.vertices, self.indices.x)
        b = dr.gather(Array2, self.vertices, self.indices.y)
        d = b - a
        self.edge_normals = dr.normalize(Array2(d.y, -d.x))
        self.normals = dr.zeros(Array2, shape=dr.width(self.vertices))
        dr.scatter_reduce(
            dr.ReduceOp.Add, self.normals, self.edge_normals, self.indices.x
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add, self.normals, self.edge_normals, self.indices.y
        )
        self.normals = dr.normalize(self.normals)

    def sdf(self, x: Array2) -> Float:
        its = self.closest_point(x)
        indices = dr.gather(Array2i, self.indices, its.prim_id)
        n1 = dr.gather(Array2, self.normals, indices.x)
        n2 = dr.gather(Array2, self.normals, indices.y)
        n = dr.lerp(n1, n2, its.t)
        return its.d * dr.sign(dr.dot(n, x - its.p))

    @dr.syntax
    def distance(self, x: Array2) -> Float:
        its = self.closest_point(x)
        return its.d

    @dr.syntax
    def distanceAD(self, x: Array2) -> Float:
        # AD version
        with dr.suspend_grad():
            its = self.closest_point(x)

        indices = dr.gather(Array2i, self.indices, its.prim_id)
        a = dr.gather(Array2, self.vertices, indices.x)
        b = dr.gather(Array2, self.vertices, indices.y)
        pa = x - a
        ba = b - a
        h = dr.clamp(dr.dot(pa, ba) / dr.dot(ba, ba), 0.0, 1.0)
        i = a + ba * h
        d = dr.norm(i - x)
        n1 = dr.gather(Array2, self.normals, indices.x)
        n2 = dr.gather(Array2, self.normals, indices.y)
        n = dr.lerp(n1, n2, its.t)
        return d * dr.sign(dr.dot(n, x - its.p))

    @dr.syntax
    def vertex_distance(self, x: Array2) -> Float:
        d_min = Float(dr.inf)
        i = Int(0)
        while i < dr.width(self.vertices):
            v = dr.gather(Array2, self.vertices, i)
            d = dr.norm(x - v)
            if d < d_min:
                d_min = d
            i += 1
        return d_min

    @dr.syntax
    def inside(self, x: Array2) -> Bool:
        return self.sdf(x) < 0

    def closest_point(self, p: Array2) -> ClosestPointRecord:
        if self.use_bvh:
            if self.bvh is None:
                print("BVH is not built. Falling back to baseline.")
                return self.closest_point_baseline(p)
            else:
                return self.bvh.closest_point(p)
        else:
            return self.closest_point_baseline(p)

    @dr.syntax
    def closest_point_baseline(self, p: Array2) -> ClosestPointRecord:
        d_min = Float(dr.inf)
        idx = Int(-1)
        i = Int(0)
        while i < dr.width(self.indices):
            f = dr.gather(Array2i, self.indices, i)
            a = dr.gather(Array2, self.vertices, f.x)
            b = dr.gather(Array2, self.vertices, f.y)
            pa = p - a
            ba = b - a
            h = dr.clamp(dr.dot(pa, ba) / dr.dot(ba, ba), 0.0, 1.0)
            # distance to the current primitive
            d = dr.norm(pa - ba * h)
            if d < d_min:
                d_min = d
                idx = i
            i += 1
        f = dr.gather(Array2i, self.indices, idx)
        a = dr.gather(Array2, self.vertices, f.x)
        b = dr.gather(Array2, self.vertices, f.y)
        pa = p - a
        ba = b - a
        h = dr.clip(dr.dot(pa, ba) / dr.dot(ba, ba), 0.0, 1.0)
        n = dr.normalize(Array2(ba.y, -ba.x))
        return ClosestPointRecord(
            valid=Bool(True), p=dr.lerp(a, b, h), n=n, t=h, d=d_min, prim_id=idx
        )

    @dr.syntax
    def closest_silhouette(
        self, x: Array2, r_max: Float = Float(dr.inf)
    ) -> ClosestSilhouettePointRecord:
        c_rec = dr.zeros(ClosestSilhouettePointRecord)
        if self.use_bvh:
            if self.snch is None:
                print("SNCH is not built. Falling back to baseline.")
                c_rec = self.closest_silhouette_baseline(x, r_max)
            else:
                c_rec = self.snch.closest_silhouette(x, r_max)
        else:
            c_rec = self.closest_silhouette_baseline(x, r_max)
        return c_rec

    @dr.syntax
    def closest_silhouette_baseline(self, x: Array2, r_max: Float = Float(dr.inf)):
        c_rec = dr.zeros(ClosestSilhouettePointRecord)
        d_min = Float(r_max)
        i = Int(0)
        while i < dr.width(self.silhouettes):
            silhouette = dr.gather(SilhouetteVertex, self.silhouettes, i)
            _c_rec = silhouette.closest_silhouette(x, r_max)
            if _c_rec.valid & (_c_rec.d < d_min):
                d_min = _c_rec.d
                c_rec = _c_rec
            i += 1
        return c_rec

    def star_radius(self, x: Array2, r_max: Float = Float(dr.inf)) -> Float:
        R = dr.norm(
            dr.maximum(dr.abs(x - self.bbox.p_min), dr.abs(x - self.bbox.p_max))
        )
        R = dr.minimum(R, r_max)
        if self.silhouettes is None:
            print("No silhouettes found. Returning max radius.")
            return R
        s_rec = self.closest_silhouette(x, R)
        R = dr.select(s_rec.valid, s_rec.d, R)
        return R

    @dr.syntax
    def star_radius_2(self, x: Array2, e: Array2, rho_max: Float = Float(1.0)) -> Float:
        R = dr.norm(
            dr.maximum(dr.abs(x - self.bbox.p_min), dr.abs(x - self.bbox.p_max))
        )
        if self.use_bvh:
            if self.snch is None:
                print("SNCH is not built. Falling back to baseline.")
                R = self.star_radius_2_3(x, e, rho_max)
            else:
                R = self.snch.star_radius_2(x, e, rho_max, r_max=R)
        else:
            R = self.star_radius_2_3(x, e, rho_max)
        return R

    @dr.syntax
    def star_radius_2_3(self, x: Array2, e: Array2, rho_max=1.0) -> Float:
        R = dr.norm(
            dr.maximum(dr.abs(x - self.bbox.p_min), dr.abs(x - self.bbox.p_max))
        )
        i = Int(0)
        while i < dr.width(self.indices):
            f = dr.gather(Array2i, self.indices, i)
            a = dr.gather(Array2, self.vertices, f.x)
            b = dr.gather(Array2, self.vertices, f.y)
            tangent = dr.normalize(b - a)
            normal = Array2(tangent.y, -tangent.x)
            dir = dr.normalize(a - x)
            if dr.dot(dir, normal) > 0.0:  # front facing
                en = dr.dot(e, normal)
                et = dr.dot(e, tangent)
                d = dr.dot(a - x, normal)
                t1 = Float(-dr.inf)
                t2 = Float(dr.inf)
                pt = x + d * normal

                # handle neumann boundary
                if dr.abs(en) > 1e-4:
                    t1 = d / en * (rho_max + et)
                    t2 = -d / en * (rho_max - et)

                ta = dr.dot(a - x, tangent)
                tb = dr.dot(b - x, tangent)
                tmin = dr.minimum(ta, tb)
                tmax = dr.maximum(ta, tb)
                if (t1 > tmin) & (t1 < tmax):
                    dist = dr.norm(pt + t1 * tangent - x)
                    R = dr.minimum(R, dist)
                if (t2 > tmin) & (t2 < tmax):
                    dist = dr.norm(pt + t2 * tangent - x)
                    R = dr.minimum(R, dist)
                if ((t1 > tmax) & (t2 > tmax)) | ((t1 < tmin) & (t2 < tmin)):
                    R = dr.minimum(R, dr.norm(a - x))
                    R = dr.minimum(R, dr.norm(b - x))
            else:  # back facing
                R = dr.minimum(R, dr.norm(a - x))
                R = dr.minimum(R, dr.norm(b - x))
            i += 1
        return R

    @dr.syntax
    def star_radius_3(self, x: Array2, u: Array2, v: Array2, rho_max=1.0) -> Float:
        i = Int(0)
        R = dr.norm(
            dr.maximum(dr.abs(x - self.bbox.p_min), dr.abs(x - self.bbox.p_max))
        )
        while i < dr.width(self.indices):
            f = dr.gather(Array2i, self.indices, i)
            pa = dr.gather(Array2, self.vertices, f.x)
            pb = dr.gather(Array2, self.vertices, f.y)
            d1 = dr.norm(pa - x)
            d2 = dr.norm(pb - x)
            tangent = dr.normalize(pb - pa)
            normal = Array2(tangent.y, -tangent.x)
            d = distance_to_plane(x, pa, normal)
            dir1 = dr.normalize(pa - x)
            dir2 = dr.normalize(pb - x)
            if dr.dot(dir1, normal) > 0:
                # front facing
                tan1 = dr.dot(dir1, tangent) / dr.dot(dir1, normal)
                tan2 = dr.dot(dir2, tangent) / dr.dot(dir2, normal)
                tan_min = dr.minimum(tan1, tan2)
                tan_max = dr.maximum(tan1, tan2)

                u_t = dr.dot(u, tangent)
                v_t = dr.dot(v, tangent)
                u_n = dr.dot(u, normal)
                v_n = dr.dot(v, normal)
                a = u_t * v_n + u_n * v_t
                b = u_t * v_t - u_n * v_n

                K_lower_bound = Float(-dr.inf)
                K_upper_bound = Float(dr.inf)

                # Neumann boundary condition
                if dr.abs(a) > 1e-4:
                    if a > 0:
                        K_lower_bound = dr.maximum(K_lower_bound, (-rho_max + b) / a)
                        K_upper_bound = dr.minimum(K_upper_bound, (rho_max + b) / a)
                    else:
                        K_lower_bound = dr.maximum(K_lower_bound, (rho_max + b) / a)
                        K_upper_bound = dr.minimum(K_upper_bound, (-rho_max + b) / a)

                if (K_lower_bound > tan_min) & (K_lower_bound < tan_max):
                    dist = dr.abs(K_lower_bound * d)
                    r = dr.sqrt(dr.sqr(d) + dr.sqr(dist))
                    R = dr.minimum(R, r)
                if (K_upper_bound > tan_min) & (K_upper_bound < tan_max):
                    dist = dr.abs(K_upper_bound * d)
                    r = dr.sqrt(dr.sqr(d) + dr.sqr(dist))
                    R = dr.minimum(R, r)
                if (tan_min > K_upper_bound) | (tan_max < K_lower_bound):
                    R = dr.minimum(R, dr.minimum(d1, d2))
            else:
                R = dr.minimum(R, dr.minimum(d1, d2))
            i += 1

        return R

    @dr.syntax
    def intersect(
        self,
        p: Array2,
        v: Array2,
        n: Array2 = Array2(0.0, 0.0),
        on_boundary: Bool = Bool(False),
        r_max: Float = Float(dr.inf),
    ) -> Intersection:
        its = dr.zeros(Intersection)
        p = dr.select(on_boundary, p - n * RayEpsilon, p)
        if self.use_bvh:
            if self.bvh is None:
                print("BVH is not built. Falling back to baseline.")
                its = self.intersect_baseline(p, v, r_max)
            else:
                its = self.bvh.intersect(p, v, r_max)
        else:
            its = self.intersect_baseline(p, v, r_max)

        if ~its.valid:
            # intersect the surface of the ball
            its = Intersection(
                valid=Bool(True),
                p=p + v * r_max,
                n=v,
                t=Float(-1.0),
                d=r_max,
                prim_id=Int(-1),
                on_boundary=Bool(False),
            )
        return its

    @dr.syntax
    def intersect_baseline(self, p: Array2, v: Array2, r_max: Float = Float(dr.inf)):
        # brute force
        d_min = Float(r_max)
        idx = Int(-1)
        is_hit = Bool(False)
        i = Int(0)
        while i < dr.width(self.indices):
            f = dr.gather(Array2i, self.indices, i)
            a = dr.gather(Array2, self.vertices, f.x)
            b = dr.gather(Array2, self.vertices, f.y)
            d = ray_intersection(p, v, a, b)
            if d < d_min:
                d_min = d
                idx = i
                is_hit = Bool(True)
            i += 1
        its = dr.zeros(Intersection)
        if is_hit:
            its = Intersection(
                valid=Bool(True),
                p=p + v * d_min,
                n=dr.gather(Array2, self.edge_normals, idx),
                t=Float(-1.0),
                d=d_min,
                prim_id=idx,
                on_boundary=Bool(True),
            )
        return its

    def sample_boundary(self, sampler: PCG32, p: Array2, r: Float):
        idx, _rnd, pmf = self.edge_pmf.sample_reuse_pmf(sampler.next_float32())
        _rnd = dr.detach(_rnd)
        face = dr.gather(Array2i, self.indices, idx)
        a = dr.gather(Array2, self.vertices, face.x)
        b = dr.gather(Array2, self.vertices, face.y)
        tangent = b - a
        normal = dr.normalize(Array2(tangent.y, -tangent.x))
        return Bool(True), BoundarySamplingRecord(
            p=dr.lerp(a, b, _rnd),
            n=normal,
            t=_rnd,
            pdf=pmf / dr.norm(tangent),
            prim_id=idx,
        )

    @dr.syntax
    def sample_boundary_bvh(self, sampler: PCG32, p: Array2, r: Float):
        return self.bvh.sample_boundary(p, r, sampler)

    @dr.syntax
    def sample_silhouette_bvh(self, sampler: PCG32, p: Array2, r: Float):
        hit, s_rec = self.silhouette_bvh.sample_boundary(p, r, sampler)
        silhouette = dr.gather(SilhouetteVertex, self.silhouettes, s_rec.prim_id)
        t1 = dr.normalize(silhouette.b - silhouette.a)
        t2 = dr.normalize(silhouette.b - silhouette.c)
        n1 = Array2(t1.y, -t1.x)
        n2 = Array2(-t2.y, t2.x)
        return hit, SilhouetteSamplingRecord(
            p=silhouette.b,
            n1=n1,
            n2=n2,
            t1=t1,
            t2=t2,
            pdf=s_rec.pdf,
            prim_id=s_rec.prim_id,
        )

    def sample_silhouette(
        self, sampler: PCG32, x: Array3, r_max: Float = Float(dr.inf)
    ):
        idx, _rnd, pmf = self.silhouette_pmf.sample_reuse_pmf(sampler.next_float32())
        silhouette = dr.gather(SilhouetteVertex, self.silhouettes, idx)
        t1 = dr.normalize(silhouette.b - silhouette.a)
        t2 = dr.normalize(silhouette.b - silhouette.c)
        n1 = Array2(t1.y, -t1.x)
        n2 = Array2(-t2.y, t2.x)
        return Bool(True), SilhouetteSamplingRecord(
            p=silhouette.b, n1=n1, n2=n2, t1=t1, t2=t2, pdf=pmf, prim_id=idx
        )

    @dr.syntax
    def closest_vertex(self, x: Array2, r_max: Float = Float(dr.inf)):
        i = Int(0)
        d_min = Float(r_max)
        idx = Int(-1)
        while i < dr.width(self.silhouettes):
            silhouette = dr.gather(SilhouetteVertex, self.silhouettes, i)
            b = silhouette.b
            d = dr.norm(b - x)
            if d < d_min:
                d_min = d
                idx = i
            i += 1

        silhouette = dr.gather(SilhouetteVertex, self.silhouettes, idx)
        t1 = dr.normalize(silhouette.b - silhouette.a)
        t2 = dr.normalize(silhouette.b - silhouette.c)
        n1 = Array2(t1.y, -t1.x)
        n2 = Array2(-t2.y, t2.x)
        T1 = dr.gather(Int, self.types, silhouette.indices[0])
        T2 = dr.gather(Int, self.types, silhouette.indices[1])
        return SilhouetteSamplingRecord(
            p=silhouette.b,
            n1=n1,
            n2=n2,
            t1=t1,
            t2=t2,
            T1=T1,
            T2=T2,
            pdf=Float(dr.inf),
            prim_id=idx,
        )

    @dr.syntax
    def off_centered_ball(self, its: ClosestPointRecord):
        p = Array2(its.p)
        # prevent super small radius
        b = dr.min(self.bbox.p_max - self.bbox.p_min) / 2.0
        a = Float(1e-3)
        i = Int(0)
        while i < 10:
            m = (a + b) / 2.0
            d = dr.abs(self.distance(p - its.n * m))
            if dr.abs(d - m) < 1e-3:  # TODO: not too small
                a = m
            else:
                b = m
            i += 1
        center = p - its.n * a
        radius = a
        return center, radius

    def J(self, prim_id):
        # for gradient
        f = dr.gather(Array2i, self.indices, prim_id)
        a = dr.gather(Array2, self.vertices, f.x)
        b = dr.gather(Array2, self.vertices, f.y)
        edge_length = dr.norm(b - a)
        return edge_length / dr.detach(edge_length)

    @dr.syntax
    def get_point(self, its: ClosestPointRecord):
        # for gradient
        t = dr.detach(its.t)
        f = dr.gather(Array2i, self.indices, its.prim_id)
        p0 = dr.gather(Array2, self.vertices, f.x)
        p1 = dr.gather(Array2, self.vertices, f.y)
        p = dr.lerp(p0, p1, t)
        J = self.J(its.prim_id)
        return ClosestPointRecord(
            p=p, n=its.n, t=t, d=Float(dr.inf), prim_id=its.prim_id, J=J
        )  # FIXME t=its.t


class Scene2D(Polyline):
    types: Int

    dirichlet_scene: Polyline = None
    neumann_scene: Polyline = None

    def __init__(self, vertices: Array2, indices: Array2i, types: Int):
        super().__init__(vertices, indices)
        self.types = types
        self.configure_subscenes()

    def configure_subscenes(self):
        # Create masks for Dirichlet and Neumann edges
        dirichlet_mask = self.types == BoundaryType.Dirichlet.value
        neumann_mask = self.types == BoundaryType.Neumann.value

        # Precompute mapping from subscene edge ids to global edge ids
        self._global_indices = dr.arange(Int, dr.width(self.indices))
        self._neumann_2_global_indices = dr.gather(
            Int, self._global_indices, dr.compress(neumann_mask)
        )
        self._dirichlet_2_global_indices = dr.gather(
            Int, self._global_indices, dr.compress(dirichlet_mask)
        )

        # Convert vertices once for vectorized processing
        vertices_np = self.vertices.numpy()

        # Build subscenes using a vectorized helper (mirrors MixedMesh)
        has_dirichlet = dr.any(dirichlet_mask)
        has_neumann = dr.any(neumann_mask)

        self.dirichlet_scene = (
            self._build_subscene_from_mask(dirichlet_mask, vertices_np)
            if has_dirichlet
            else None
        )

        if has_neumann:
            # Keep original-indexed edge list for sampling back on the full scene
            self.neumann_indices = dr.gather(
                Array2i, self.indices, dr.compress(neumann_mask)
            )
            self.neumann_scene = self._build_subscene_from_mask(
                neumann_mask, vertices_np
            )
        else:
            self.neumann_scene = None

    def _build_subscene_from_mask(self, edge_mask, vertices_np):
        has_edges = dr.any(edge_mask)
        if not has_edges:
            return None

        # Select edges matching the mask (shape: 2 x E)
        selected_indices = dr.gather(Array2i, self.indices, dr.compress(edge_mask))
        edges_np = selected_indices.numpy()

        # Compute unique vertex set and remapping
        flat = edges_np.reshape(-1, order="F")
        unique_vertices, inverse = np.unique(flat, return_inverse=True)
        new_indices_np = inverse.reshape(edges_np.shape, order="F").astype(np.int32)

        # Gather corresponding vertices
        new_vertices_np = vertices_np[:, unique_vertices]

        # Construct the sub-polyline
        return Polyline(
            vertices=Array2(new_vertices_np), indices=Array2i(new_indices_np)
        )

    @dr.syntax
    def sample_neumann_boundary(self, sampler: PCG32, p: Array2, r: Float):
        idx, _rnd, pmf = self.neumann_scene.edge_pmf.sample_reuse_pmf(
            sampler.next_float32()
        )
        _rnd = dr.detach(_rnd)
        face = dr.gather(Array2i, self.neumann_indices, idx)
        a = dr.gather(Array2, self.vertices, face.x)
        b = dr.gather(Array2, self.vertices, face.y)
        tangent = b - a
        normal = dr.normalize(Array2(tangent.y, -tangent.x))
        return Bool(True), BoundarySamplingRecord(
            p=dr.lerp(a, b, _rnd),
            n=normal,
            t=_rnd,
            pdf=pmf / dr.norm(tangent),
            prim_id=dr.gather(Int, self._neumann_2_global_indices, idx),
            type=Int(BoundaryType.Neumann.value),
        )
