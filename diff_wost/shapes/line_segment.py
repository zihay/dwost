from dataclasses import dataclass

from diff_wost.core.fwd import PCG32, Array2, Array3, Bool, Float, Int, dr
from diff_wost.core.math import closest_point_line_segment
from diff_wost.render.interaction import (
    BoundarySamplingRecord3D,
    ClosestPointRecord,
    ClosestPointRecord3D,
    Intersection,
    Intersection3D,
)


@dataclass
class LineSegment:
    a: Array2
    b: Array2
    index: Int  # index in the original array
    sorted_index: Int  # index in the sorted array
    type: Int = Int(-1)

    @dr.syntax
    def normal(self):
        t = self.b - self.a
        return dr.normalize(Array2(t.y, -t.x))

    @dr.syntax
    def length(self):
        return dr.norm(self.b - self.a)

    @dr.syntax
    def surface_area(self):
        return dr.norm(self.b - self.a)

    @dr.syntax
    def sample_point(self, sampler: PCG32):
        t = sampler.next_float32()
        p = self.a + (self.b - self.a) * t
        pdf = 1.0 / dr.norm(self.b - self.a)
        return p, pdf, t

    @dr.syntax
    def sphere_intersect(self, x: Array2, R: Float):
        d, _, _ = closest_point_line_segment(x, self.a, self.b)
        return d <= R

    @dr.syntax
    def _ray_intersect_nonwatertight(self, x: Array2, d: Array2, r_max: Float):
        """
        Initial ray-segment intersection test. 
        """
        its = dr.zeros(Intersection)
        u = self.a - x
        v = self.b - self.a
        dv = d[0] * v[1] - d[1] * v[0]
        if dr.abs(dv) > dr.epsilon(Float):
            # not parallel
            ud = u[0] * d[1] - u[1] * d[0]
            s = ud / dv
            if (s >= 0.0) & (s <= 1.0):
                uv = u[0] * v[1] - u[1] * v[0]
                t = uv / dv

                if (t >= 0.0) & (t <= r_max):
                    its = Intersection(
                        valid=Bool(True),
                        p=x + d * t,
                        n=self.normal(),
                        t=s,
                        d=dr.abs(t),
                        prim_id=self.index,
                        on_boundary=Bool(True),
                        type=self.type,
                    )
        return its

    @dr.syntax
    def ray_intersect_watertight(self, x: Array2, d: Array2, r_max: Float):
        """
        Watertight ray-segment intersection test using shear and permutation, adapted from
        "Watertight Ray/Triangle Intersection" by Woop, Benthin, and Wald (2013).
        """
        its = dr.zeros(Intersection)

        # 1. Translate endpoints to ray origin
        Ax = self.a.x - x.x
        Ay = self.a.y - x.y
        Bx = self.b.x - x.x
        By = self.b.y - x.y

        # 2. Permute dimensions: pick dominant axis to maximise precision
        #    and avoid dividing by a near-zero component
        u_A = Float(0.0)
        v_A = Float(0.0)
        u_B = Float(0.0)
        v_B = Float(0.0)
        Du = Float(0.0)
        Dv = Float(0.0)
        if dr.abs(d.x) > dr.abs(d.y):
            u_A = Ay
            v_A = Ax
            u_B = By
            v_B = Bx
            Du = d.y
            Dv = d.x
        else:
            u_A = Ax
            v_A = Ay
            u_B = Bx
            v_B = By
            Du = d.x
            Dv = d.y

        # 3. Shear-transform endpoints into ray-aligned space (ray passes through u=0)
        shear = -Du / Dv
        Au = u_A + v_A * shear
        Bu = u_B + v_B * shear

        # 4. Watertight straddling test (half-open interval):
        #    endpoints must lie on strictly opposite sides of u=0.
        #    A vertex shared by two adjacent segments will straddle in exactly one.
        if (Au < 0.0) != (Bu < 0.0):
            Av = v_A / Dv
            Bv = v_B / Dv
            denom = Bu - Au  # non-zero: Au and Bu have opposite signs
            inv_denom = Float(1.0) / denom
            t_val = (Av * Bu - Bv * Au) * inv_denom
            s_val = -Au * inv_denom  # barycentric (0 at a, 1 at b)

            if (t_val >= 0.0) & (t_val <= r_max):
                its = Intersection(
                    valid=Bool(True),
                    p=x + d * t_val,
                    n=self.normal(),
                    t=s_val,
                    d=dr.abs(t_val),
                    prim_id=self.index,
                    on_boundary=Bool(True),
                    type=self.type,
                )
        return its
    

    def ray_intersect(self, x: Array2, d: Array2, r_max: Float, watertight: bool = True):
        if watertight:
            return self.ray_intersect_watertight(x, d, r_max)
        else:
            return self._ray_intersect_nonwatertight(x, d, r_max)

    @dr.syntax
    def closest_point(self, p: Array2):
        pa = p - self.a
        ba = self.b - self.a
        t = dr.normalize(ba)
        n = Array2(t.y, -t.x)
        h = dr.clamp(dr.dot(pa, ba) / dr.dot(ba, ba), 0.0, 1.0)
        i = self.a + ba * h
        d = dr.norm(i - p)
        return ClosestPointRecord(
            valid=Bool(True), p=i, n=n, t=h, d=d, prim_id=self.index, type=self.type
        )

    @dr.syntax
    def bounding_point(
        self,
        x: Array2,
        e: Array2,
        rho_max: Float = Float(1.0),
        r_max: Float = Float(dr.inf),
    ):
        c_rec = dr.zeros(ClosestPointRecord)
        R = Float(r_max)
        a = self.a
        b = self.b
        tangent = dr.normalize(b - a)
        normal = Array2(tangent.y, -tangent.x)
        dir = dr.normalize(a - x)
        if dr.dot(dir, normal) > 0.0:
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
                if dist < R:
                    c_rec.valid = Bool(True)
                    c_rec.p = pt + t1 * tangent
                    c_rec.t = t1
                    c_rec.d = dist
                    R = dist
            if (t2 > tmin) & (t2 < tmax):
                dist = dr.norm(pt + t2 * tangent - x)
                if dist < R:
                    c_rec.valid = Bool(True)
                    c_rec.p = pt + t2 * tangent
                    c_rec.t = t2
                    c_rec.d = dist
                    R = dist
            if ((t1 > tmax) & (t2 > tmax)) | ((t1 < tmin) & (t2 < tmin)):
                dist = dr.norm(a - x)
                if dist < R:
                    c_rec.valid = Bool(True)
                    c_rec.p = a
                    c_rec.t = Float(0.0)
                    c_rec.d = dist
                    R = dist
                dist = dr.norm(b - x)
                if dist < R:
                    c_rec.valid = Bool(True)
                    c_rec.p = b
                    c_rec.t = Float(1.0)
                    c_rec.d = dist
                    R = dist
        else:
            dist = dr.norm(a - x)
            if dist < R:
                c_rec.valid = Bool(True)
                c_rec.p = a
                c_rec.t = Float(0.0)
                c_rec.d = dist
                R = dist
            dist = dr.norm(b - x)
            if dist < R:
                c_rec.valid = Bool(True)
                c_rec.p = b
                c_rec.t = Float(1.0)
                c_rec.d = dist
                R = dist
        c_rec.prim_id = self.sorted_index
        c_rec.type = self.type
        return c_rec

    @dr.syntax
    def star_radius_2(
        self,
        x: Array2,
        e: Array2,
        rho_max: Float = Float(1.0),
        r_max: Float = Float(dr.inf),
    ):
        R = Float(r_max)
        a = self.a
        b = self.b
        tangent = dr.normalize(b - a)
        normal = Array2(tangent.y, -tangent.x)
        dir = dr.normalize(a - x)
        if dr.dot(dir, normal) > 0.0:
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
        else:
            R = dr.minimum(R, dr.norm(a - x))
            R = dr.minimum(R, dr.norm(b - x))
        return R


@dataclass
class LineSegment3D:
    a: Array3
    b: Array3
    index: Int  # index in the original array
    sorted_index: Int  # index in the sorted array
    type: Int = Int(-1)

    @dr.syntax
    def direction(self):
        return self.b - self.a

    @dr.syntax
    def length(self):
        return dr.norm(self.b - self.a)

    @dr.syntax
    def surface_area(self):
        # for sampling PMF/PDF weighting, use length as area analogue for segments
        return dr.norm(self.b - self.a)

    @dr.syntax
    def sample_point(self, sampler: PCG32):
        t = sampler.next_float32()
        p = self.a + (self.b - self.a) * t
        pdf = 1.0 / dr.maximum(self.length(), dr.epsilon(Float))
        # no canonical normal; return zero and let caller handle
        return BoundarySamplingRecord3D(
            p=p,
            n=Array3(0.0, 0.0, 0.0),
            uv=Array2(t, 0.0),
            pdf=pdf,
            prim_id=self.index,
            type=self.type,
        )

    @dr.syntax
    def sphere_intersect(self, x: Array3, R: Float):
        # check if closest point on the segment lies within radius R
        ba = self.b - self.a
        pa = x - self.a
        denom = dr.maximum(dr.dot(ba, ba), dr.epsilon(Float))
        t = dr.clamp(dr.dot(pa, ba) / denom, 0.0, 1.0)
        closest = self.a + ba * t
        return dr.norm(closest - x) <= R

    @dr.syntax
    def ray_intersect(self, x: Array3, d: Array3, r_max: Float):
        # Ray-line segment intersection in 3D is a measure-zero event; we consider
        # no direct intersection (only triangles are intersectable). Return invalid.
        return dr.zeros(Intersection3D)

    @dr.syntax
    def closest_point(self, p: Array3):
        pa = p - self.a
        ba = self.b - self.a
        denom = dr.maximum(dr.dot(ba, ba), dr.epsilon(Float))
        h = dr.clamp(dr.dot(pa, ba) / denom, 0.0, 1.0)
        i = self.a + ba * h
        d = dr.norm(i - p)
        # no canonical normal for a line; return zero vector
        return ClosestPointRecord3D(
            valid=Bool(True),
            p=i,
            n=Array3(0.0, 0.0, 0.0),
            uv=Array2(h, 0.0),
            d=d,
            prim_id=self.index,
            type=self.type,
        )
