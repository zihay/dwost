from dataclasses import dataclass

from diff_wost.core.fwd import PCG32, Array3, Bool, Float, Int, dr
from diff_wost.core.math import uniform_on_sphere
from diff_wost.render.greens_fn_3d import GreensBall3D
from diff_wost.shapes.mesh import Mesh
from diff_wost.solvers.solver import Solver


@dataclass
class WoS3D(Solver):
    control_variates: bool = True
    antithetic_variates: bool = True

    @dr.syntax
    def u(self, x: Array3, scene: Mesh, sampler: PCG32):
        p = Array3(x)
        result = Float(0)
        i = Int(0)
        active = Bool(True)
        while (i < self.nsteps) & active:
            d_dirichlet = dr.abs(scene.distance(p))
            # shrink size for numerical stability

            if d_dirichlet < self.eps:
                c_rec = scene.closest_point(p)
                result += scene.g(c_rec)
                active = Bool(False)
            else:
                dir = uniform_on_sphere(sampler)
                p = p + dir * d_dirichlet

            i += 1
        return result

    @dr.syntax
    def dude(
        self,
        p: Array3,
        e: Array3,
        scene: Mesh,
        sampler: PCG32,
        on_boundary: Bool = Bool(False),
        n: Array3 = Array3(0.0, 0.0, 0.0),
    ):
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
            dir = uniform_on_sphere(sampler)

            # source term
            dir_src = uniform_on_sphere(sampler)
            greens = GreensBall3D(c=p, R=R)
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
                _result2 = 3.0 / R * _u * dr.dot(e, dir)
                result += _result2
                i += 1
                iter += 1

        return result / self.nwalks
