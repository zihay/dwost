from dataclasses import dataclass
from diff_wost.core.fwd import *


@dataclass
class GreensBall3D:
    c: Array3
    R: Float

    def G(self, x: Array3, y: Array3) -> Float:
        r = dr.norm(y - x)
        return -(1. / r - self.R / (self.R * self.R - dr.dot(x - self.c, y - self.c)))/(4. * dr.pi)

    @dr.syntax
    def evaluate(self, r: Float):
        return (1. / r - 1. / self.R) / (4. * dr.pi)

    @dr.syntax
    def _sample(self, v: Array2, sampler: PCG32):
        # sample radius r from pdf 6.0f * r * (R - r) / R^3 using Ulrich's polar method
        u1 = sampler.next_float32()
        u2 = sampler.next_float32()
        phi = 2.0 * dr.pi * u2
        r = (1.0 + dr.sqrt(1.0 - dr.cbrt(u1 * u1)) * dr.cos(phi)) * self.R / 2.0
        r = dr.maximum(r, 1e-6)
        if r > self.R:
            r = self.R / 2.0
        pdf = self.evaluate(r) / self.norm()

        return self.c + r * v, r, pdf

    def norm(self):
        # norm of the Greens function (over the ball with radius R)
        return self.R * self.R / 6.0

    def gradient_norm(self, r: Float):
        r3 = r * r * r
        return (1. / r3 - 1. / (self.R * self.R * self.R)) / (4. * dr.pi)

    def gradient(self, r: Float, y: Array3):
        d = y - self.c
        return d * self.gradient_norm(r)

    def sample_r_pdf(self, v: Array3, sampler: PCG32):
        p, r, pdf = self._sample(v, sampler)
        return p, r, pdf

    def directionSampledPoissonKernel(self, y: Array3):
        return Float(1.0)

    def poissonKernel(self):
        return 1. / (4. * dr.pi)

    def poissonKernelGradient(self, y: Array3):
        d = y - self.c
        return 3. * d / (4. * dr.pi * self.R * self.R)
