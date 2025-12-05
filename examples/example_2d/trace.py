from dataclasses import dataclass

from diff_wost.core.fwd import PCG32, Array2, Bool, Float, Int, dr
from diff_wost.core.math import sample_tea_32
from diff_wost.shapes.polyline import Scene2D
from diff_wost.solvers.wost_grad import WoStGrad


@dataclass
class StreamlineTracer:
    scene: Scene2D
    solver: WoStGrad
    step_size: Float = Float(0.1)
    sampler: PCG32 = None
    nwalks: int = 100

    def __post_init__(self):
        pass

    def next_position(self, p: Array2, seed=0):
        npoints = dr.width(p)
        nsamples = npoints * self.nwalks
        result = dr.zeros(Array2, npoints)
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

    @dr.syntax
    def _next_position(self, p: Array2, sampler: PCG32):
        current_p = Array2(p)
        y = Array2(current_p)
        done = Bool(False)
        if self.should_increment(current_p):
            current_p = self.increment_streamline(current_p, sampler)
            y_tilde = Array2(current_p)
            if self.should_increment(current_p):
                fy = y_tilde - y
                fy_tilde = current_p - y_tilde
                current_p = y + (fy + fy_tilde) * 0.5
            else:
                current_p = Array2(y_tilde)
                done = Bool(True)
        return current_p

    @dr.syntax
    def should_increment(self, p: Array2):
        ret = Bool(True)
        d = self.scene.sdf(p)
        if d > 0.0:
            ret = Bool(False)
        return ret

    @dr.syntax
    def increment_streamline(self, p: Array2, sampler: PCG32):
        dudx = self.solver.walk_dude(p, Array2(1.0, 0.0), self.scene, sampler)
        dudy = self.solver.walk_dude(p, Array2(0.0, 1.0), self.scene, sampler)
        grad = Array2(dudx, dudy)
        return p + grad * self.step_size
