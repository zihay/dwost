from dataclasses import dataclass

from diff_wost.core.fwd import PCG32, Array2, Array3, Bool, Float, Int, dr
from diff_wost.core.math import sample_tea_32, uniform_on_sphere
from diff_wost.render.greens_fn_3d import GreensBall3D
from diff_wost.render.interaction import Intersection3D
from diff_wost.shapes.mesh import Scene3D
from diff_wost.solvers.solver import Solver


@dataclass
class WoSt3D(Solver):
    control_variates: bool = False
    antithetic_variates: bool = False
    min_R: float = 1e-3

    @dr.syntax
    def handle_source(
        self,
        scene: Scene3D,
        x: Array3,
        dir: Array3,
        R: Float,
        r_max: Float,
        sampler: PCG32,
    ):
        greens = GreensBall3D(c=x, R=R)
        result = Float(0.0)
        p, r, pdf = greens.sample_r_pdf(dir, sampler)
        if dr.norm(p - x) < r_max:
            result += greens.norm() * scene.f(p)
        return result

    def solve_dude(self, p, e, scene, seed=0):
        npoints = dr.width(p)
        # multiply the wavefront size by nwalks
        idx = dr.arange(Int, npoints)
        v0, v1 = sample_tea_32(seed, idx)
        dr.eval(v0, v1)
        sampler = PCG32(size=npoints, initstate=v0, initseq=v1)
        return self.dude(p, e, scene, sampler)

    @dr.syntax
    def u(
        self,
        x: Array3,
        scene: Scene3D,
        sampler: PCG32,
        on_boundary: Bool = Bool(False),
        n: Array3 = Array3(0.0, 0.0, 0.0),
    ):
        its = Intersection3D(
            valid=Bool(True),
            p=Array3(x),
            n=Array3(n),
            uv=Array2(0.0, 0.0),
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
            p = dr.select(its.on_boundary, its.p - its.n * 1e-3, its.p)
            star_radius = scene.star_radius(p, r_max=d_dirichlet) * 0.98
            # prevent getting stuck in the concave region
            star_radius = dr.maximum(star_radius, self.min_R)
            R = dr.minimum(R, star_radius)

            if d_dirichlet < self.eps:
                c_rec = scene.dirichlet_scene.closest_point(its.p)
                result += scene.g(c_rec)
                active = Bool(False)
            else:
                # compute the single-sample Neumann contribution
                if scene.neumann_scene is not None:
                    b_hit, b_rec = scene.neumann_scene.sample_boundary(
                        sampler, p=its.p, r_max=R
                    )
                    if b_hit:
                        G = GreensBall3D(c=its.p, R=R).G(its.p, b_rec.p)
                        G = dr.clip(G, -100.0, 100.0)
                        alpha = dr.select(its.on_boundary, 2.0, 1.0)
                        if dr.norm(its.p - b_rec.p) <= R:
                            h = scene.h(b_rec)
                            result -= G * h * alpha / b_rec.pdf

                dir = uniform_on_sphere(sampler)
                if its.on_boundary & (dr.dot(dir, its.n) > 0):
                    dir = -dir
                _p = its.p  # save for source term
                its = scene.intersect(
                    its.p, dir, n=its.n, on_boundary=its.on_boundary, r_max=R
                )

                result += self.handle_source(
                    scene, _p, dir, R, r_max=its.d, sampler=sampler
                )

            i += 1
        return result

    @dr.syntax
    def dude(self, p: Array3, e: Array3, scene: Scene3D, sampler: PCG32):
        self.results = dr.zeros(Float, self.nwalks)
        result = Float(0.0)
        i = Int(0)
        R = dr.abs(scene.distance(p))
        # control variates
        u_mean = Float(0.0)
        u_count = Float(0)
        src_mean = Float(0.0)
        src_count = Float(0)

        active = scene.inside(p)
        while (i < self.nwalks) & active:
            dir = uniform_on_sphere(sampler)

            greens = GreensBall3D(c=p, R=R)
            greens_norm = greens.norm()
            p_src, r_src, pdf_src = greens.sample_r_pdf(dir, sampler)

            niters = 2 if self.antithetic_variates else 1
            state = sampler.state + 0  # save sampler state
            iter = Int(0)
            while (iter < niters) & (i < self.nwalks):
                # handle source term
                if iter == 1:
                    dir = -dir
                    p_src = 2.0 * p - p_src
                src = greens_norm * scene.f(p_src)
                src_grad = dr.dot(greens.gradient(r_src, p_src), e) / (
                    pdf_src * greens_norm
                )
                src_mean = src_mean * (src_count / (src_count + 1)) + src / (
                    src_count + 1
                )
                src_count += 1
                _src = src
                if self.control_variates:
                    _src = src - src_mean

                _result1 = _src * src_grad
                result += _result1

                # handle next step
                if iter == 1:
                    dir = -dir
                _p = p + dir * R
                sampler.state = state + 0  # reset sampler state
                u = self.u(_p, scene, sampler)
                if dr.isnan(u) | dr.isinf(u) | (dr.abs(u) > 10000.0):
                    u = Float(0.0)

                u_mean = u_mean * (u_count / (u_count + 1)) + u / (u_count + 1)
                u_count += 1
                _u = u
                if self.control_variates:
                    _u = u - u_mean
                _result2 = 3.0 / R * _u * dr.dot(e, dir)
                result += _result2
                i += 1
                iter += 1

        if dr.isnan(result) | dr.isinf(result) | (dr.abs(result) > 10000.0):
            result = Float(0.0)

        # result = dr.clip(result, -1000.0, 1000.0)

        return result / self.nwalks
