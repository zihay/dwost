from dataclasses import dataclass

from diff_wost.core.fwd import PCG32, Array2, Bool, Float, Int, dr
from diff_wost.core.math import uniform_angle, uniform_in_disk
from diff_wost.render.greens_fn import GreensBall
from diff_wost.render.interaction import (
    BoundarySamplingRecord,
    ClosestPointRecord,
    Intersection,
    SilhouetteSamplingRecord,
)
from diff_wost.shapes.polyline import Scene2D
from diff_wost.solvers.solver import Solver
from diff_wost.solvers.wost import WoSt


@dataclass
class WoStGrad(Solver):
    ignore_neumann: bool = False
    ignore_boundary: bool = False
    ignore_silhouette: bool = False
    ignore_source: bool = False
    control_variates: bool = True
    russian_roulette: bool = False
    min_R: Float = Float(1e-2)
    rho_max: Float = Float(1.2)

    @dr.syntax
    def handle_boundary(
        self,
        scene: Scene2D,
        x: Array2,
        e: Array2,
        R: Float,
        b_rec: BoundarySamplingRecord,
    ):
        result = Float(0)
        G = GreensBall(c=x, R=R).G(x, b_rec.p)
        G = dr.clip(G, -100.0, 100.0)
        if dr.norm(x - b_rec.p) <= R:
            normal = b_rec.n
            tangent = Array2(-normal.y, normal.x)
            e_t = dr.dot(e, tangent)
            e_n = dr.dot(e, normal)

            # Neumann boundary condition
            f_tn = scene.dhdt(b_rec)
            result += e_t * f_tn - e_n * scene.f(b_rec.p)

        return -G * result / b_rec.pdf

    @dr.syntax
    def handle_silhouette(
        self,
        scene: Scene2D,
        x: Array2,
        e: Array2,
        R: Float,
        s_rec: SilhouetteSamplingRecord,
    ):
        result = Float(0)
        G = GreensBall(c=x, R=R).G(x, s_rec.p)
        G = dr.clip(G, -100.0, 100.0)
        if dr.norm(s_rec.p - x) <= R:
            # Neumann boundary condition
            h1 = scene.h(BoundarySamplingRecord(p=s_rec.p, n=s_rec.n1))
            h2 = scene.h(BoundarySamplingRecord(p=s_rec.p, n=s_rec.n2))
            result -= h1 * dr.dot(e, s_rec.t1)
            result -= h2 * dr.dot(e, s_rec.t2)
        return -G * result / s_rec.pdf

    @dr.syntax
    def handle_source(
        self,
        scene: Scene2D,
        x: Array2,
        e: Array2,
        dir: Array2,
        R: Float,
        r_max: Float,
        sampler: PCG32,
    ):
        result = Float(0.0)
        greens = GreensBall(c=x, R=R)
        p = greens.sample(dir, sampler)
        if dr.norm(p - x) < r_max:
            result += greens.norm() * dr.dot(scene.dfdx(p), e)
        return result

    @dr.syntax
    def dude(
        self,
        x: Array2,
        e: Array2,
        scene: Scene2D,
        sampler: PCG32,
        on_boundary: Bool = Bool(False),
        n: Array2 = Array2(0.0, 0.0),
    ):
        its = Intersection(
            valid=Bool(True),
            p=Array2(x),
            n=Array2(n),
            t=Float(0),
            d=Float(dr.inf),
            prim_id=Int(-1),
            on_boundary=Bool(on_boundary),
        )

        result = Float(0)
        i = Int(0)
        active = its.on_boundary | scene.inside(its.p)
        throughput = Float(1.0)

        while (i < self.nsteps) & active:
            inflate = Bool(False)
            d_dirichlet = Float(dr.inf)
            if scene.dirichlet_scene is not None:
                d_dirichlet = dr.abs(scene.dirichlet_scene.distance(its.p))

            if d_dirichlet < self.eps:
                c_rec = scene.closest_point(its.p)
                normal = c_rec.n
                tangent = Array2(-normal.y, normal.x)
                dudt = scene.dgdt(c_rec)
                dudn = self.dudn(c_rec, scene, sampler)
                # # accumulate the tangent component from the Dirichlet boundary
                result += throughput * dudt * dr.dot(e, tangent)
                result += throughput * dudn * dr.dot(e, normal)
                active = Bool(False)
            else:
                R = d_dirichlet
                if scene.neumann_scene is not None:
                    d_star_radius = scene.neumann_scene.star_radius_2(
                        its.p, e, rho_max=Float(self.rho_max)
                    )
                    R = dr.minimum(R, d_star_radius)
                if its.on_boundary:
                    if R < self.min_R:
                        inflate = Bool(True)
                    R = dr.maximum(R, self.min_R)

                alpha = dr.select(its.on_boundary, 2.0, 1.0)

                if scene.neumann_scene is not None:
                    if not self.ignore_boundary:
                        b_hit, b_rec = scene.neumann_scene.sample_boundary_bvh(
                            sampler, its.p, R
                        )
                        if b_hit:
                            result += (
                                throughput
                                * alpha
                                * self.handle_boundary(scene, its.p, e, R, b_rec)
                            )

                    if not self.ignore_silhouette:
                        s_hit, s_rec = scene.neumann_scene.sample_silhouette_bvh(
                            sampler, its.p, R
                        )
                        if ~inflate & s_hit:
                            result += (
                                throughput
                                * alpha
                                * self.handle_silhouette(scene, its.p, e, R, s_rec)
                            )

                theta = uniform_angle(sampler)
                dir = Array2(dr.cos(theta), dr.sin(theta))
                if its.on_boundary & (dr.dot(dir, its.n) > 0):
                    dir = -dir

                _p = Array2(its.p)  # save for source estimate
                its: Intersection = scene.intersect(
                    its.p, dir, n=its.n, on_boundary=its.on_boundary, r_max=R
                )

                if not self.ignore_source:
                    result += throughput * self.handle_source(
                        scene, _p, e, dir, R, dr.abs(its.d), sampler
                    )

                if its.on_boundary:
                    normal = its.n
                    tangent = Array2(-normal.y, normal.x)
                    e_t = dr.dot(e, tangent)
                    e_n = dr.dot(e, normal)
                    K = dr.dot(dir, tangent) / dr.dot(dir, normal)
                    # Neumann boundary condition
                    result += throughput * e_n * scene.h(its)
                    throughput *= e_t - K * e_n
                    e = tangent

                # Russian roulette
                if dr.abs(throughput) < 1.0:
                    if sampler.next_float32() > dr.abs(throughput):
                        active = Bool(False)
                    else:
                        throughput /= dr.abs(throughput)
            i += 1

            if dr.isnan(result) | dr.isinf(result) | (dr.abs(result) > 1000.0):
                result = Float(0.0)

        return result

    @dr.syntax
    def dudn(self, its: ClosestPointRecord, scene: Scene2D, sampler: PCG32):
        dudn = Float(0)
        c, R = scene.off_centered_ball(its)

        theta = uniform_angle(sampler)
        theta = dr.clip(theta, 3e-1, 2 * dr.pi - 3e-1)
        # forward direction
        f_dir = its.n
        # perpendicular direction
        p_dir = Array2(-f_dir.y, f_dir.x)
        # sample a point on the largest ball
        dir = Array2(f_dir * dr.cos(theta) + p_dir * dr.sin(theta))
        p = c + R * dir
        u_ref = scene.g(its)
        # start wost to estimate u
        solver = WoSt(nsteps=200, eps=1e-4)
        u = solver.u(p, scene, sampler)

        # prevent leaking
        if scene.neumann_scene is not None:
            d_dirichlet = scene.dirichlet_scene.distance(p)
            d_neumann = scene.neumann_scene.distance(p)
            if (d_dirichlet < 2e-2) & (d_neumann < 2e-2):
                u = u_ref

        P = 1.0 / (dr.cos(theta) - 1.0)
        # control variate
        if self.control_variates:
            dudn += P * (u - u_ref) / R
        else:
            dudn += P * u / R

        # source term
        p_src = c + R * uniform_in_disk(sampler)
        f_ref = scene.f(its.p)
        f = scene.f(p_src)
        x2 = dr.squared_norm(its.p - c)
        y2 = dr.squared_norm(p_src - c)
        P = (
            (x2 - y2) / dr.maximum(dr.squared_norm(its.p - p_src), 1e-6) * R / 2
        )  # pdf included

        if self.control_variates:
            dudn -= (f - f_ref) * P + f_ref * R / 2.0  # NOTE the minus sign
        else:
            dudn -= f * P
        return dudn
