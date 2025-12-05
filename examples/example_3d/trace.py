from dataclasses import dataclass

from diff_wost.core.fwd import PCG32, Array3, Bool, Float, Int, dr
from diff_wost.core.math import sample_tea_32
from diff_wost.shapes.mesh import Scene3D
from diff_wost.solvers.wost_grad_3d import WoStGrad3D


@dataclass
class StreamlineTracer:
    scene: Scene3D
    solver: WoStGrad3D
    step_size: Float = Float(0.1)
    sampler: PCG32 = None
    nwalks: int = 100

    def __post_init__(self):
        pass

    def next_position(self, p: Array3, seed=0):
        npoints = dr.width(p)
        nsamples = npoints * self.nwalks
        result = dr.zeros(Array3, npoints)
        # multiply the wavefront size by nwalks
        idx = dr.arange(Int, nsamples)
        _p = dr.repeat(p, self.nwalks)
        if self.nwalks > 1:
            idx //= self.nwalks
        v0, v1 = sample_tea_32(seed, idx)
        dr.eval(v0, v1)
        sampler = PCG32(size=nsamples, initstate=v0, initseq=v1)
        value = self._next_position(_p, sampler)
        dr.scatter_reduce(dr.ReduceOp.Add, result, value, idx)
        if self.nwalks > 1:
            result /= self.nwalks
        return result

    # @dr.syntax
    def _next_position(self, p: Array3, sampler: PCG32):
        current_p = self.increment_streamline(p, sampler)
        return current_p

    @dr.syntax
    def should_increment(self, p: Array3):
        ret = Bool(True)
        d = self.scene.sdf(p)
        if d > 1e-5:
            ret = Bool(False)
        return ret

    @dr.syntax
    def increment_streamline(self, p: Array3, sampler: PCG32):
        dudx = self.solver.walk_dude(p, Array3(1.0, 0.0, 0.0), self.scene, sampler)
        dr.eval(dudx, sampler)
        dudy = self.solver.walk_dude(p, Array3(0.0, 1.0, 0.0), self.scene, sampler)
        dr.eval(dudy, sampler)
        dudz = self.solver.walk_dude(p, Array3(0.0, 0.0, 1.0), self.scene, sampler)
        dr.eval(dudz, sampler)
        grad = Array3(dudx, dudy, dudz)
        return p + grad * self.step_size
