import numpy as np

from diff_wost.core.distr import DiscreteDistribution
from diff_wost.core.fwd import (
    PCG32,
    Array2,
    Array2i,
    Array3,
    Array3i,
    Array4i,
    Bool,
    Float,
    Int,
    RayEpsilon,
    dr,
)
from diff_wost.core.math import (
    closest_point_triangle,
    interpolate,
    ray_triangle_intersect,
)
from diff_wost.render.interaction import (
    BoundarySamplingRecord3D,
    ClosestPointRecord3D,
    ClosestSilhouettePointRecord3D,
    Intersection3D,
    SilhouetteSamplingRecord3D,
)
from diff_wost.shapes.bvh3d import BoundingBox3D, Segment3DBVH
from diff_wost.shapes.polyline import BoundaryType
from diff_wost.shapes.silhouette_edge import SilhouetteEdge
from diff_wost.shapes.snch3d import SNCH3D
from diff_wost.shapes.triangle import Triangle


class Mesh:
    vertices: Array3
    indices: Array3i

    normals: Array3 = None
    face_normals: Array3 = None
    areas: Float = None

    silhouettes: SilhouetteEdge = None

    edges: Array2i = None
    f2e: Array3i = None  # face to edge
    e2f: Array2i = None  # edge to face

    # bvh: BVH3D = None
    snch: SNCH3D = None
    silhouette_bvh: Segment3DBVH = None

    use_bvh: bool = True
    use_snch: bool = True
    use_silhouette: bool = True
    use_f2e_e2f: bool = True

    def g(self, c_rec: ClosestPointRecord3D):
        return self.u(c_rec.p)

    def h(self, b_rec: BoundarySamplingRecord3D):
        return dr.dot(b_rec.n, self.dudx(b_rec.p))

    def __init__(
        self,
        vertices: Array3,
        indices: Array3i,
        use_bvh: bool = True,
        use_snch: bool = True,
        use_silhouette: bool = True,
        use_f2e_e2f: bool = True,
    ):
        self.vertices = vertices
        self.indices = indices
        self.use_bvh = use_bvh
        self.use_snch = use_snch
        self.use_silhouette = use_silhouette
        self.use_f2e_e2f = use_f2e_e2f
        self.configure()

    def configure(self):
        # compute bounding box
        self.bbox = BoundingBox3D(
            p_min=Array3(
                dr.min(self.vertices.x),
                dr.min(self.vertices.y),
                dr.min(self.vertices.z),
            ),
            p_max=Array3(
                dr.max(self.vertices.x),
                dr.max(self.vertices.y),
                dr.max(self.vertices.z),
            ),
        )

        self.configure_normal()
        self.configure_face_pmf()
        if self.use_silhouette:
            self.configure_silhouette()
        if self.use_f2e_e2f:
            self.configure_f2e_e2f()

        # if self.use_bvh:
        # self.bvh = BVH3D(vertices=self.vertices, indices=self.indices)
        if self.use_snch:
            self.snch = SNCH3D(vertices=self.vertices, indices=self.indices)

    def configure_normal(self):
        # polulate normals
        a = dr.gather(Array3, self.vertices, self.indices.x)
        b = dr.gather(Array3, self.vertices, self.indices.y)
        c = dr.gather(Array3, self.vertices, self.indices.z)
        u = b - a
        v = c - a
        self.face_normals = dr.normalize(dr.cross(u, v))
        self.normals = dr.zeros(Array3, shape=dr.width(self.vertices))
        dr.scatter_reduce(
            dr.ReduceOp.Add, self.normals, self.face_normals, self.indices.x
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add, self.normals, self.face_normals, self.indices.y
        )
        dr.scatter_reduce(
            dr.ReduceOp.Add, self.normals, self.face_normals, self.indices.z
        )
        self.normals = dr.normalize(self.normals)

    def configure_silhouette(self):
        """
        Vectorized construction of silhouette edges.

        A silhouette edge is an undirected edge shared by exactly two triangles.
        This builds the edge list and adjacency without Python loops by grouping
        canonical edge pairs and collecting opposite vertices per face.
        """
        # Convert DrJit buffers to NumPy arrays once for vectorized processing
        vertices_np = self.vertices.numpy()  # (3, V): vertex positions
        faces_np = self.indices.numpy()  # (3, F): per-face vertex indices

        # Number of faces in the mesh
        F = faces_np.shape[1]
        # Early out when there are no faces (empty mesh)
        if F == 0:
            self.silhouettes = None
            return

        # Vertex index arrays for the three corners of each face
        a01 = faces_np[0]  # first  vertex per face
        b01 = faces_np[1]  # second vertex per face
        c01 = faces_np[2]  # third  vertex per face
        # Build the three undirected edges (u,v) per face in canonical order (u<=v)
        # along with the opposite vertex index and owning face index
        e0_u = np.minimum(a01, b01).astype(np.int32)  # edge (a,b) - smaller index first
        e0_v = np.maximum(a01, b01).astype(
            np.int32
        )  # edge (a,b) - larger  index second
        e0_opp = c01.astype(np.int32)  # opposite vertex to (a,b) is c
        e0_face = np.arange(F, dtype=np.int32)  # face indices [0..F-1]

        e1_u = np.minimum(b01, c01).astype(np.int32)  # edge (b,c)
        e1_v = np.maximum(b01, c01).astype(np.int32)
        e1_opp = a01.astype(np.int32)  # opposite vertex to (b,c) is a
        e1_face = e0_face  # same face indices

        e2_u = np.minimum(c01, a01).astype(np.int32)  # edge (c,a)
        e2_v = np.maximum(c01, a01).astype(np.int32)
        e2_opp = b01.astype(np.int32)  # opposite vertex to (c,a) is b
        e2_face = e0_face  # same face indices

        # Concatenate the three edges per face into one long edge list
        u = np.concatenate([e0_u, e1_u, e2_u])  # edge endpoints (u)
        v = np.concatenate([e0_v, e1_v, e2_v])  # edge endpoints (v)
        opp = np.concatenate([e0_opp, e1_opp, e2_opp])  # opposite vertices
        face = np.concatenate([e0_face, e1_face, e2_face])  # owning face ids

        # Group identical undirected edges using a canonical integer key
        Vcount = vertices_np.shape[1]  # number of vertices (for key hashing)
        key = u.astype(np.int64) * np.int64(Vcount) + v.astype(
            np.int64
        )  # unique key per (u,v)
        order = np.argsort(
            key, kind="mergesort"
        )  # stable sort to keep paired edges adjacent
        u_s = u[order]  # sorted u endpoints
        v_s = v[order]  # sorted v endpoints
        opp_s = opp[order]  # sorted opposite vertices
        face_s = face[order]  # sorted owning face ids
        key_s = key[order]  # sorted keys

        # Compute unique keys and how many times each appears (edge multiplicity)
        uniq_keys, first_idx, counts = np.unique(
            key_s, return_index=True, return_counts=True
        )
        # Keep only edges shared by exactly two faces
        valid = counts == 2
        if not np.any(valid):
            self.silhouettes = None
            return

        # Indices of the two entries (two half-edges) for each valid edge
        first = first_idx[valid]
        second = first + 1

        # Endpoints (u,v), opposite vertices (c,d), and adjacent face ids (f1,f2)
        ua = u_s[first]
        vb = v_s[first]
        oc = opp_s[first]
        od = opp_s[second]
        f1 = face_s[first]
        f2 = face_s[second]

        # Gather geometry
        A = vertices_np[:, ua]
        B = vertices_np[:, vb]
        C = vertices_np[:, oc]
        D = vertices_np[:, od]

        # Build silhouette edge array
        silhouettes = SilhouetteEdge(
            a=Array3(A),
            b=Array3(B),
            c=Array3(C),
            d=Array3(D),
            indices=Array4i(np.vstack([oc, ua, vb, od]).astype(np.int32)),
            index=Int(np.arange(ua.shape[0], dtype=np.int32)),
            face_indices=Array2i(np.vstack([f1, f2]).astype(np.int32)),
        )

        # Filter flat edges
        n0 = silhouettes.n0()
        n1 = silhouettes.n1()
        flatness = dr.abs(dr.dot(n0, n1))
        self.silhouettes = dr.gather(
            SilhouetteEdge, silhouettes, dr.compress(flatness < (1.0 - 1e-5))
        )

        # Length-weighted sampling pmf
        edge_lengths = dr.norm(self.silhouettes.a - self.silhouettes.b)
        self.silhouette_pmf = DiscreteDistribution(edge_lengths)
        if self.use_bvh:
            _vertices = self.vertices
            _indices = Array2i(silhouettes.indices[1], silhouettes.indices[2])
            self.silhouette_bvh = Segment3DBVH(_vertices, _indices)

    def configure_f2e_e2f(self):
        edges = {}
        face_to_edge = []
        edge_to_face = {}

        indices = self.indices.numpy().T

        for i in range(indices.shape[0]):
            f = indices[i]
            a, b, c = f[0], f[1], f[2]

            edge_indices = []
            for edge in [(a, b), (b, c), (c, a)]:
                edge = tuple(sorted(edge))  # sort edge to make it unique
                if edge not in edges:
                    edges[edge] = len(edges)
                    edge_to_face[edge] = [-1, -1]

                edge_indices.append(edges[edge])

                if edge_to_face[edge][0] == -1:
                    edge_to_face[edge][0] = i
                elif edge_to_face[edge][1] == -1:
                    edge_to_face[edge][1] = i

            face_to_edge.append(edge_indices)

        self.f2e = Array3i(np.array(face_to_edge).T)
        self.e2f = Array2i(
            np.array([edge_to_face[edge] for edge in sorted(edges, key=edges.get)]).T
        )
        self.edges = Array2i(np.array(sorted(edges.keys(), key=edges.get)).T)

    def configure_face_pmf(self):
        a = dr.gather(Array3, self.vertices, self.indices.x)
        b = dr.gather(Array3, self.vertices, self.indices.y)
        c = dr.gather(Array3, self.vertices, self.indices.z)
        self.face_pmf = DiscreteDistribution(dr.norm(dr.cross(b - a, c - a)))

    @dr.syntax
    def distance(self, x: Array3) -> Float:
        c_rec = self.closest_point(x)
        return c_rec.d

    @dr.syntax
    def sdf(self, x: Array3) -> Float:
        c_rec = self.closest_point(x)
        return c_rec.d * dr.sign(dr.dot(c_rec.n, x - c_rec.p))

    @dr.syntax
    def inside(self, x: Array3) -> Bool:
        return self.sdf(x) < 0

    @dr.syntax
    def closest_point(self, x: Array3) -> ClosestPointRecord3D:
        its = dr.zeros(ClosestPointRecord3D)
        if self.use_bvh:
            if self.snch is None:
                print("BVH is not built, using baseline method")
                its = self.closest_point_baseline(x)
            else:
                its = self.snch.closest_point(x)
        else:
            its = self.closest_point_baseline(x)
        return its

    @dr.syntax
    def closest_point_baseline(self, x: Array3) -> ClosestPointRecord3D:
        d_min = Float(dr.inf)
        idx = Int(-1)
        i = Int(0)
        while i < dr.width(self.indices):
            f = dr.gather(Array3i, self.indices, i)
            a = dr.gather(Array3, self.vertices, f.x)
            b = dr.gather(Array3, self.vertices, f.y)
            c = dr.gather(Array3, self.vertices, f.z)
            pt, uv, d = closest_point_triangle(x, a, b, c)
            if d < d_min:
                d_min = d
                idx = i
            i += 1
        f = dr.gather(Array3i, self.indices, idx)
        a = dr.gather(Array3, self.vertices, f.x)
        b = dr.gather(Array3, self.vertices, f.y)
        c = dr.gather(Array3, self.vertices, f.z)
        pt, uv, d = closest_point_triangle(x, a, b, c)
        fn = dr.gather(Array3, self.face_normals, idx)
        return ClosestPointRecord3D(
            valid=Bool(True), p=pt, n=fn, uv=uv, d=d_min, prim_id=idx
        )

    def star_radius(self, x: Array3, r_max: Float = Float(dr.inf)) -> Float:
        r_bound = dr.norm(
            dr.maximum(dr.abs(x - self.bbox.p_min), dr.abs(x - self.bbox.p_max))
        )
        R = dr.minimum(r_bound, r_max)
        if self.silhouettes is None:
            return R
        s_rec = self.closest_silhouette(x, R)
        R = dr.select(s_rec.valid, s_rec.d, R)
        return R

    @dr.syntax
    def star_radius_2(
        self, x: Array3, e: Array3, rho_max=Float(1.01), r_max=Float(dr.inf)
    ) -> Float:
        R = self.star_radius(x, r_max)
        if self.use_bvh:
            if self.snch is None:
                print("BVH is not built, using baseline method")
                R = self.star_radius_2_baseline(x, e, rho_max, R)
            else:
                R = self.snch.star_radius_3(x, e, rho_max, R)
        else:
            R = self.star_radius_2_baseline(x, e, rho_max, R)
        return R

    @dr.syntax
    def star_radius_2_baseline(
        self, x: Array3, e: Array3, rho_max=Float(1.0), r_max=Float(dr.inf)
    ) -> Float:
        R = dr.norm(
            dr.maximum(dr.abs(x - self.bbox.p_min), dr.abs(x - self.bbox.p_max))
        )
        R = dr.minimum(R, r_max)
        i = Int(0)
        while i < dr.width(self.indices):
            f = dr.gather(Array3i, self.indices, i)
            a = dr.gather(Array3, self.vertices, f.x)
            b = dr.gather(Array3, self.vertices, f.y)
            c = dr.gather(Array3, self.vertices, f.z)
            triangle = Triangle(a, b, c, i, i)
            _R = triangle.star_radius_2(x, e, rho_max, R)
            R = dr.minimum(R, _R)
            i += 1
        return R

    @dr.syntax
    def intersect(
        self,
        p: Array3,
        v: Array3,
        n: Array3 = Array3(0.0, 0.0, 0.0),
        on_boundary: Bool = Bool(False),
        r_max: Float = Float(dr.inf),
    ):
        its = dr.zeros(Intersection3D)
        # TODO: do not use if statement, it will cause segmentation fault in the bvh
        p = dr.select(on_boundary, p - n * RayEpsilon, p)
        if self.use_bvh:
            if self.snch is None:
                print("BVH is not built, using baseline method")
                its = self.intersect_baseline(p, v, r_max)
            else:
                its = self.snch.intersect(p, v, r_max)
        else:
            its = self.intersect_baseline(p, v, r_max)

        if ~its.valid:
            # intersect the surface of the ball
            its = Intersection3D(
                valid=Bool(True),
                p=p + v * r_max,
                n=v,
                uv=Array2(0.0, 0.0),
                d=r_max,
                prim_id=Int(-1),
                on_boundary=Bool(False),
            )
        return its

    @dr.syntax
    def intersect_all(self, p: Array3, v: Array3, r_max: Float = Float(dr.inf)):
        its = dr.zeros(Intersection3D)
        hits = Int(0)
        active = Bool(True)
        while active:
            p = dr.select(its.on_boundary, its.p + v * 1e-5, its.p)
            its = self.intersect(p, v, r_max)
            if ~its.valid:
                active = Bool(False)
            else:
                hits += 1

        active = Bool(True)
        while active:
            p = dr.select(its.on_boundary, its.p - v * 1e-5, its.p)
            its = self.intersect(p, -v, r_max)
            if ~its.valid:
                active = Bool(False)
            else:
                hits += 1
        return hits

    @dr.syntax
    def intersect_baseline(self, p: Array3, v: Array3, r_max: Float = Float(dr.inf)):
        d_min = Float(r_max)
        uv = Array2(0.0, 0.0)
        idx = Int(-1)
        is_hit = Bool(False)
        i = Int(0)

        while i < dr.width(self.indices):
            f = dr.gather(Array3i, self.indices, i)
            a = dr.gather(Array3, self.vertices, f.x)
            b = dr.gather(Array3, self.vertices, f.y)
            c = dr.gather(Array3, self.vertices, f.z)
            _its = ray_triangle_intersect(p, v, a, b, c)
            if _its.valid & (_its.d < d_min):
                d_min = _its.d
                uv = Array2(_its.uv.x, _its.uv.y)
                idx = i
                is_hit = Bool(True)
            i += 1
        its = Intersection3D(
            valid=Bool(True),
            p=p + v * d_min,
            n=Array3(0.0, 0.0, 0.0),
            uv=Array2(0.0, 0.0),
            d=d_min,
            prim_id=idx,
            on_boundary=is_hit,
        )
        if is_hit:
            its.n = dr.gather(Array3, self.face_normals, idx)
            its.uv = uv
        return its

    def sample_boundary(self, sampler: PCG32, p: Array3, r_max: Float = Float(dr.inf)):
        idx, _rnd, pmf = self.face_pmf.sample_reuse_pmf(sampler.next_float32())
        f = dr.gather(Array3i, self.indices, idx)
        a = dr.gather(Array3, self.vertices, f.x)
        b = dr.gather(Array3, self.vertices, f.y)
        c = dr.gather(Array3, self.vertices, f.z)
        area = dr.norm(dr.cross(b - a, c - a)) / 2.0

        r1 = Float(sampler.next_float32())
        r2 = Float(sampler.next_float32())
        u = dr.sqrt(r1) * (1 - r2)
        v = dr.sqrt(r1) * r2
        p = interpolate(a, b, c, Array2(u, v))
        n = dr.normalize(dr.cross(b - a, c - a))
        return Bool(True), BoundarySamplingRecord3D(
            p=p, n=n, uv=Array2(u, v), pdf=pmf / area, prim_id=idx
        )

    def sample_boundary_bvh(
        self, sampler: PCG32, p: Array3, r_max: Float = Float(dr.inf)
    ):
        return self.snch.sample_boundary(p, r_max, sampler)

    def sample_silhouette(
        self, sampler: PCG32, x: Array3, r_max: Float = Float(dr.inf)
    ):
        if self.silhouettes is None:
            return Bool(False), dr.zeros(SilhouetteSamplingRecord3D)
        idx, _rnd, pmf = self.silhouette_pmf.sample_reuse_pmf(sampler.next_float32())
        silhouette = dr.gather(SilhouetteEdge, self.silhouettes, idx)
        length = dr.norm(silhouette.b - silhouette.a)
        n1 = dr.normalize(
            dr.cross(silhouette.b - silhouette.a, silhouette.c - silhouette.a)
        )
        n2 = dr.normalize(
            dr.cross(silhouette.d - silhouette.a, silhouette.b - silhouette.a)
        )
        t1 = dr.normalize(dr.cross(silhouette.b - silhouette.a, n1))
        t2 = dr.normalize(dr.cross(silhouette.a - silhouette.b, n2))
        p = dr.lerp(silhouette.a, silhouette.b, Float(sampler.next_float32()))
        return Bool(True), SilhouetteSamplingRecord3D(
            p=p, n1=n1, n2=n2, t1=t1, t2=t2, pdf=pmf / length, prim_id=idx
        )

    @dr.syntax
    def sample_silhouette_bvh(
        self, sampler: PCG32, p: Array3, r_max: Float = Float(dr.inf)
    ):
        s_rec = dr.zeros(SilhouetteSamplingRecord3D)
        hit, b_rec = self.silhouette_bvh.sample_boundary(p, r_max, sampler)
        if hit:
            silhouette = dr.gather(SilhouetteEdge, self.silhouettes, b_rec.prim_id)
            n1 = dr.normalize(
                dr.cross(silhouette.b - silhouette.a, silhouette.c - silhouette.a)
            )
            n2 = dr.normalize(
                dr.cross(silhouette.d - silhouette.a, silhouette.b - silhouette.a)
            )
            t1 = dr.normalize(dr.cross(silhouette.b - silhouette.a, n1))
            t2 = dr.normalize(dr.cross(silhouette.a - silhouette.b, n2))
            s_rec = SilhouetteSamplingRecord3D(
                p=b_rec.p,
                n1=n1,
                n2=n2,
                t1=t1,
                t2=t2,
                pdf=b_rec.pdf,
                prim_id=b_rec.prim_id,
            )
        return hit, s_rec

    @dr.syntax
    def closest_silhouette(self, p: Array3, r_max: Float = Float(dr.inf)):
        c_rec = dr.zeros(ClosestSilhouettePointRecord3D)
        if self.use_bvh:
            if self.snch is None:
                print("BVH is not built, using baseline method")
                c_rec = self.closest_silhouette_baseline(p, r_max)
            else:
                c_rec = self.snch.closest_silhouette(p, r_max)
        else:
            c_rec = self.closest_silhouette_baseline(p, r_max)
        return c_rec

    @dr.syntax
    def closest_silhouette_baseline(self, p: Array3, r_max: Float = Float(dr.inf)):
        i = Int(0)
        d_min = Float(r_max)
        c_rec = dr.zeros(ClosestSilhouettePointRecord3D)
        while i < dr.width(self.silhouettes):
            silhouette = dr.gather(SilhouetteEdge, self.silhouettes, i)
            _c_rec = silhouette.closest_silhouette(p, r_max)
            if _c_rec.valid & (_c_rec.d < d_min):
                d_min = _c_rec.d
                c_rec = _c_rec
            i += 1
        return c_rec

    @dr.syntax
    def off_centered_ball(self, its: ClosestPointRecord3D):
        p = Array3(its.p)
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


class Scene3D(Mesh):
    types: Int

    dirichlet_scene: Mesh = None
    neumann_scene: Mesh = None

    def __init__(self, vertices: Array3, indices: Array3i, types: Int):
        super().__init__(vertices, indices)
        self.types = types
        self.configure_subscenes()

    def configure_subscenes(self):
        # Masks for Dirichlet and Neumann faces
        dirichlet_mask = self.types == BoundaryType.Dirichlet.value
        neumann_mask = self.types == BoundaryType.Neumann.value

        # Convert vertices to NumPy once
        vertices_np = self.vertices.numpy()

        # Build subscenes using a vectorized helper
        self.dirichlet_scene = self._build_subscene_from_mask(
            dirichlet_mask, vertices_np
        )
        self.neumann_scene = self._build_subscene_from_mask(neumann_mask, vertices_np)

    def _build_subscene_from_mask(self, face_mask, vertices_np):
        has_faces = dr.any(face_mask)
        if not has_faces:
            return None

        # Select face indices matching the mask (shape: 3 x F)
        selected_indices = dr.gather(Array3i, self.indices, dr.compress(face_mask))
        faces_np = selected_indices.numpy()

        # Compute unique vertex set and remapping without Python loops
        flat = faces_np.reshape(-1, order="F")
        unique_vertices, inverse = np.unique(flat, return_inverse=True)
        new_indices_np = inverse.reshape(faces_np.shape, order="F").astype(np.int32)

        # Gather corresponding vertices
        new_vertices_np = vertices_np[:, unique_vertices]

        # Construct the sub-mesh
        return Mesh(vertices=Array3(new_vertices_np), indices=Array3i(new_indices_np))
