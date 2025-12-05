from dataclasses import dataclass

from diff_wost.core.fwd import PCG32, Array2, Array3, Bool, Float, Int, dr
from diff_wost.core.math import uniform_on_sphere
from diff_wost.render.greens_fn_3d import GreensBall3D
from diff_wost.render.interaction import (
    BoundarySamplingRecord3D,
    Intersection3D,
    SilhouetteSamplingRecord3D,
)
from diff_wost.shapes.mesh import Mesh, Scene3D
from diff_wost.solvers.solver import Solver


@dataclass
class WoStGrad3D(Solver):
    ignore_source: bool = False
    ignore_silhouette: bool = False
    ignore_boundary: bool = False
    control_variates: bool = True
    rho_max: Float = Float(1.2)
    min_R: Float = Float(1e-1)

    @dr.syntax
    def handle_boundary(
        self,
        scene: Mesh,
        x: Array3,
        e: Array3,
        R: Float,
        b_rec: BoundarySamplingRecord3D,
        ref: Array3 = Array3(0.0, 0.0, 0.0),
    ):
        result = Float(0)
        G = GreensBall3D(c=x, R=R).G(x, b_rec.p)
        G = dr.clip(G, -100.0, 100.0)  # prevent large values
        if dr.norm(x - b_rec.p) <= R:
            normal = b_rec.n
            e_n = dr.dot(e, normal)
            e_t = e - normal * e_n

            # Neumann boundary condition
            f_tn = dr.dot(scene.hessian(b_rec.p) @ e_t, normal)
            result += f_tn - e_n * scene.f(b_rec.p)
            if self.control_variates:
                result -= dr.dot(ref, normal)
        return -G * result / b_rec.pdf

    @dr.syntax
    def handle_silhouette(
        self,
        scene: Mesh,
        x: Array3,
        e: Array3,
        R: Float,
        s_rec: SilhouetteSamplingRecord3D,
        ref: Array3 = Array3(0.0, 0.0, 0.0),
    ):
        result = Float(0)
        G = GreensBall3D(c=x, R=R).G(x, s_rec.p)
        G = dr.clip(G, -100.0, 100.0)  # prevent large values
        if dr.norm(s_rec.p - x) <= R:
            h1 = dr.dot(scene.dudx(s_rec.p), s_rec.n1)
            h2 = dr.dot(scene.dudx(s_rec.p), s_rec.n2)
            e1 = dr.dot(e, s_rec.t1)
            e2 = dr.dot(e, s_rec.t2)
            if self.control_variates:
                h1 = h1 - dr.dot(ref, s_rec.n1)
                h2 = h2 - dr.dot(ref, s_rec.n2)
            result -= h1 * e1
            result -= h2 * e2
        return -G * result / s_rec.pdf

    @dr.syntax
    def handle_source(
        self,
        scene: Mesh,
        x: Array3,
        e: Array3,
        dir: Array3,
        R: Float,
        r_max: Float,
        sampler: PCG32,
    ):
        greens = GreensBall3D(c=x, R=R)
        result = Float(0.0)
        p, r, pdf = greens.sample_r_pdf(dir, sampler)
        if dr.norm(p - x) < r_max:
            result += greens.norm() * dr.dot(scene.dfdx(p), e)
        return result

    @dr.syntax
    def dude(
        self,
        x: Array3,
        e: Array3,
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
        active = scene.inside(x)

        i = Int(0)
        while (i < self.nsteps) & active:
            d_dirichlet = Float(dr.inf)
            inflate = Bool(False)
            if scene.dirichlet_scene is not None:
                d_dirichlet = dr.abs(scene.dirichlet_scene.distance(its.p))

            if d_dirichlet < self.eps:
                c_rec = scene.dirichlet_scene.closest_point(its.p)
                normal = c_rec.n
                e_n = dr.dot(e, normal)
                e_t = e - normal * e_n
                dudn = self.dudn(c_rec, scene, sampler)
                result += dr.dot(scene.dudx(c_rec.p), e_t)
                result += dudn * e_n
                active = Bool(False)
            else:
                R = d_dirichlet
                if scene.neumann_scene is not None:
                    its.p = dr.select(its.on_boundary, its.p - its.n * 1e-3, its.p)
                    star_radius = scene.neumann_scene.star_radius_2(
                        its.p, e, rho_max=Float(self.rho_max)
                    )
                    R = dr.minimum(R, star_radius)
                    if its.on_boundary:  # prevent large reflectance
                        if R < self.min_R:
                            inflate = Bool(True)
                        R = dr.maximum(R, self.min_R)

                alpha = dr.select(its.on_boundary, 2.0, 1.0)

                b_ref = Array3(0.0, 0.0, 0.0)
                s_ref = Array3(0.0, 0.0, 0.0)

                if scene.neumann_scene is not None:
                    b_hit, b_rec = scene.neumann_scene.sample_boundary(
                        sampler, its.p, R
                    )
                    if ~inflate & b_hit:
                        result += alpha * self.handle_boundary(
                            scene, its.p, e, R, b_rec, b_ref
                        )

                    s_hit, s_rec = scene.neumann_scene.sample_silhouette(
                        sampler, its.p, R
                    )
                    if ~inflate & s_hit:
                        result += alpha * self.handle_silhouette(
                            scene, its.p, e, R, s_rec, s_ref
                        )

                dir = uniform_on_sphere(sampler)
                if its.on_boundary & (dr.dot(dir, its.n) > 0):
                    dir = -dir
                _p = Array3(its.p)  # save for source term
                its = scene.intersect(
                    its.p, dir, n=its.n, on_boundary=its.on_boundary, r_max=R
                )

                # boundary control variates
                result += dr.dot(_p - its.p, b_ref)

                if not self.ignore_source:
                    result += self.handle_source(scene, _p, e, dir, R, its.d, sampler)

                if its.on_boundary:
                    e_n = dr.dot(e, its.n)
                    e_t = e - its.n * e_n
                    dir_n = dr.dot(dir, its.n)
                    dir_t = dir - dir_n * its.n

                    # silhouette control variates
                    ref_t = s_ref - dr.dot(s_ref, its.n) * its.n
                    result += dr.dot(dir_t, ref_t) * e_n / dir_n

                    # Neumann boundary condition
                    result += e_n * scene.h(its)
                    e = e_t - e_n * dir_t / dir_n

                # Russian roulette
                if dr.norm(e) < 1.0:
                    if dr.norm(e) < sampler.next_float32():
                        active = Bool(False)
                    else:
                        e /= dr.norm(e)

            i += 1

            if dr.isnan(result) | dr.isinf(result) | (dr.abs(result) > 1000.0):
                result = Float(0.0)

        return result
