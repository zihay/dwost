from dataclasses import dataclass

from diff_wost.core.fwd import PCG32, Array2, Bool, Float, Int, dr
from diff_wost.core.math import sample_tea_32, uniform_angle
from diff_wost.render.greens_fn import GreensBall
from diff_wost.render.interaction import (
    BoundarySamplingRecord,
    ClosestPointRecord,
    Intersection,
)
from diff_wost.shapes.polyline import Scene2D
from diff_wost.solvers.solver import Solver


@dataclass
class WoSt(Solver):
    control_variates: bool = True
    antithetic_variates: bool = True
    min_R: float = 1e-3

    def solve_dude(self, p, e, scene, seed=0):
        npoints = dr.width(p)
        # multiply the wavefront size by nwalks
        idx = dr.arange(Int, npoints)
        v0, v1 = sample_tea_32(seed, idx)
        dr.eval(v0, v1)
        sampler = PCG32(size=npoints, initstate=v0, initseq=v1)
        return self.dude(p, e, scene, sampler)

    @dr.syntax
    def dude(
        self,
        p: Array2,
        e: Array2,
        scene: Scene2D,
        sampler: PCG32,
        on_boundary: Bool = Bool(False),
        n: Array2 = Array2(0.0, 0.0),
    ):
        self.results = dr.zeros(Float, self.nwalks)
        result = Float(0.0)
        i = Int(0)
        R = dr.abs(scene.distance(p))
        # control variates
        u_sum = Float(0.0)
        u_count = Float(0)
        src_sum = Float(0.0)
        src_count = Float(0)

        active = scene.inside(p)
        while (i < self.nwalks) & active:
            theta = uniform_angle(sampler)
            dir = Array2(dr.cos(theta), dr.sin(theta))

            theta_src = uniform_angle(sampler)
            dir_src = Array2(dr.cos(theta_src), dr.sin(theta_src))
            greens = GreensBall(c=p, R=R)
            greens_norm = greens.norm()
            p_src, r_src, pdf_src = greens.sample_r_pdf(dir_src, sampler)

            niters = 2 if self.antithetic_variates else 1
            state = sampler.state + 0  # save sampler state
            iter = Int(0)
            while (iter < niters) & (i < self.nwalks):
                # handle source term
                if iter == 1:
                    dir_src = -dir_src
                    p_src = 2.0 * p - p_src
                src = greens_norm * scene.f(p_src)
                src_grad = dr.dot(greens.gradient(r_src, p_src), e) / (
                    pdf_src * greens_norm
                )
                src_sum += src
                src_count += 1
                _src = src
                if self.control_variates:
                    _src = src - (src_sum / src_count)

                _result1 = _src * src_grad
                result += _result1

                # handle next step
                if iter == 1:
                    dir = -dir
                _p = p + dir * R
                sampler.state = state + 0  # reset sampler state
                u = self.u(_p, scene, sampler)
                u_sum += u
                u_count += 1
                _u = u
                if self.control_variates:
                    _u = u - u_sum / u_count
                _result2 = 2.0 / R * _u * dr.dot(e, dir)

                if (dr.abs(_result2) > 1e6) | dr.isnan(_result2) | dr.isinf(_result2):
                    _result2 = Float(0.0)

                result += _result2
                i += 1
                iter += 1

        return result / self.nwalks

    @dr.syntax
    def handle_boundary(
        self,
        scene: Scene2D,
        p: Array2,
        R: Float,
        b_rec: BoundarySamplingRecord,
        ref: Array2 = Array2(0.0, 0.0),
    ):
        result = Float(0)
        G = GreensBall(c=p, R=R).G(p, b_rec.p)
        G = dr.clip(G, -10.0, 10.0)
        if dr.norm(p - b_rec.p) <= R:
            h = scene.h(b_rec)
            if self.control_variates:
                h = h - dr.dot(ref, b_rec.n)
            result -= G * h / b_rec.pdf
        return result

    @dr.syntax
    def boundary_control_variates(self, scene: Scene2D, p: Array2):
        c_rec: ClosestPointRecord = scene.closest_point(p)
        return scene.h(c_rec) * c_rec.n

    @dr.syntax
    def handle_source(
        self,
        scene: Scene2D,
        x: Array2,
        dir: Array2,  # sample direction
        R: Float,  # radius of the ball
        r_max: Float,  # distance to the boundary
        sampler: PCG32,
    ):
        result = Float(0.0)
        greens = GreensBall(c=x, R=R)
        p = greens.sample(dir, sampler)
        if dr.norm(p - x) < r_max:
            result += greens.norm() * scene.f(p)
        return result

    @dr.syntax
    def u(
        self,
        x: Array2,
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
        active = Bool(True)

        while (i < self.nsteps) & active:
            R = d_dirichlet = Float(dr.inf)
            if scene.dirichlet_scene is not None:
                R = d_dirichlet = dr.abs(scene.dirichlet_scene.distance(its.p))
            # shrink size for numerical stability
            if scene.neumann_scene is not None:
                p = dr.select(its.on_boundary, its.p - its.n * 1e-5, its.p)  # important
                star_radius = (
                    scene.neumann_scene.star_radius(p, r_max=d_dirichlet) * 0.98
                )
                # prevent getting stuck in the concave region
                star_radius = dr.maximum(star_radius, self.min_R)
                R = dr.minimum(R, star_radius)

            if d_dirichlet < self.eps:
                c_rec = scene.dirichlet_scene.closest_point(its.p)
                result += scene.g(c_rec)
                active = Bool(False)
            else:
                ref = Array2(0.0, 0.0)
                if scene.neumann_scene is not None:
                    b_hit, b_rec = scene.neumann_scene.sample_boundary(
                        sampler, p=its.p, r=Float(dr.inf)
                    )

                    ref = self.boundary_control_variates(scene, its.p)

                    if b_hit:
                        alpha = dr.select(its.on_boundary, 2.0, 1.0)
                        result += alpha * self.handle_boundary(
                            scene, its.p, R, b_rec, ref
                        )

                theta = uniform_angle(sampler)
                dir = Array2(dr.cos(theta), dr.sin(theta))
                if its.on_boundary & (dr.dot(dir, its.n) > 0):
                    dir = -dir
                _p = Array2(its.p)

                its: Intersection = scene.intersect(
                    its.p, dir, n=its.n, on_boundary=its.on_boundary, r_max=R
                )

                if self.control_variates:
                    result += dr.dot(_p - its.p, ref)

                result += self.handle_source(scene, _p, dir, R, its.d, sampler)

            i += 1

        if dr.isnan(result) | dr.isinf(result) | (dr.abs(result) > 100.0):
            result = Float(0.0)

        return result
