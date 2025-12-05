from dataclasses import dataclass
import numpy as np

from diff_wost.core.fwd import *
from diff_wost.core.math import uniform_on_sphere
from diff_wost.render.greens_fn import GreensBall


class YukawaGreensFnBall3D:
    c: Array3 = Array3(0.0, 0.0, 0.0)
    R: Float = Float(1.0)
    lambda_val: Float = Float(1.0)

    def __init__(self, c: Array3, R: float, lambda_val: float = 1.0):
        self.c = c
        self.R = R
        self.lambda_val = lambda_val
        self.sqrtLambda = dr.sqrt(self.lambda_val)
        self.muR = self.R * self.sqrtLambda
        self.expmuR = dr.exp(-self.muR)
        exp2muR = self.expmuR * self.expmuR
        coshmuR = (1.0 + exp2muR) / (2.0 * self.expmuR)
        self.sinhmuR = (1.0 - exp2muR) / (2.0 * self.expmuR)
        self.K32muR = self.expmuR * (1.0 + 1.0 / self.muR)
        self.I32muR = coshmuR - self.sinhmuR / self.muR

    # Evaluates the Green's function
    @dr.syntax
    def evaluate(self, r: Float) -> Float:
        mur = r * self.sqrtLambda
        expmur = dr.exp(-mur)
        exp2mur = expmur * expmur
        sinhmur = (1.0 - exp2mur) / (2.0 * expmur)
        return (expmur - self.expmuR * sinhmur / self.sinhmuR) / (4.0 * dr.pi * r)

    def G(self, x: Array3, y: Array3) -> Float:
        r1 = dr.maximum(1e-4, dr.norm(y - x))
        r2 = (self.R*self.R - dr.dot(x - self.c, y - self.c))/self.R
        mur1 = r1*self.sqrtLambda
        mur2 = r2*self.sqrtLambda
        expmur1 = dr.exp(-mur1)
        expmur2 = dr.exp(-mur2)
        sinhmur1 = (1.0 - expmur1*expmur1)/(2.0*expmur1)
        sinhmur2 = (1.0 - expmur2*expmur2)/(2.0*expmur2)
        Q1 = (expmur1 - self.expmuR*sinhmur1/self.sinhmuR)/r1
        Q2 = (expmur2 - self.expmuR*sinhmur2/self.sinhmuR)/r2

        return -(Q1 - Q2)/(4.0*dr.pi)

    # Evaluates the norm of the Green's function
    @dr.syntax
    def norm(self) -> Float:
        return (1.0 - 4.0 * dr.pi * self.poissonKernel()) / self.lambda_val

    # Evaluates the gradient norm of the Green's function
    @dr.syntax
    def gradient_norm(self, r: Float) -> Float:
        r2 = r * r
        mur = r * self.sqrtLambda
        expmur = dr.exp(-mur)
        exp2mur = expmur * expmur
        coshmur = (1.0 + exp2mur) / (2.0 * expmur)
        sinhmur = (1.0 - exp2mur) / (2.0 * expmur)
        K32mur = expmur * (1.0 + 1.0 / mur)
        I32mur = coshmur - sinhmur / mur
        Qr = self.sqrtLambda * (K32mur - I32mur * self.K32muR / self.I32muR)

        return Qr / (4.0 * dr.pi * r2)

    # Evaluates the gradient of the Green's function
    @dr.syntax
    def gradient(self, r: Float, y: Array3) -> Array3:
        d = y - self.c
        return d * self.gradient_norm(r)

    # Samples a point on the surface of the ball
    @dr.syntax
    def sample_surface(self, sampler: PCG32) -> tuple:
        # Generate a random direction on unit sphere and scale by radius
        dir = uniform_on_sphere(sampler)
        y = self.c + self.R * dir
        pdf = 1.0 / (4.0 * dr.pi)

        return y, pdf

    # Evaluates the Poisson Kernel (normal derivative of the Green's function)
    @dr.syntax
    def poissonKernel(self) -> Float:
        return self.muR / (4.0 * dr.pi * self.sinhmuR)

    # Evaluates the radial dampening factor associated with the centered Poisson Kernel
    @dr.syntax
    def poissonKernelDampeningFactor(self, r: Float) -> Float:
        mur = r * self.sqrtLambda
        expmur = dr.exp(-mur)
        exp2mur = expmur * expmur
        coshmur = (1.0 + exp2mur) / (2.0 * expmur)
        sinhmur = (1.0 - exp2mur) / (2.0 * expmur)
        K32mur = expmur * (1.0 + 1.0 / mur)
        I32mur = coshmur - sinhmur / mur
        Q = K32mur + I32mur * self.expmuR / self.sinhmuR

        return mur * Q

    # Evaluates the centered Poisson Kernel at a point y with normal n
    def P(self, x: Array3, y: Array3, n: Array3) -> Float:
        xy = y - self.c
        xyNorm = dr.norm(xy)
        r = dr.maximum(1e-4, xyNorm)
        r2 = r * r
        xy = xy / xyNorm

        return self.poissonKernelDampeningFactor(r) * dr.dot(n, xy) / (4.0 * dr.pi * r2)

    # Directly evaluates the centered Poisson Kernel at a point y over the direction sampling pdf
    @dr.syntax
    def directionSampledPoissonKernel(self, y: Array3) -> Float:
        r = dr.maximum(1e-4, dr.norm(y - self.c))
        return self.poissonKernelDampeningFactor(r)

    # Evaluates the gradient of the Poisson Kernel
    @dr.syntax
    def poissonKernelGradient(self, y: Array3) -> Array3:
        d = y - self.c
        QR = self.lambda_val / self.I32muR

        return d * QR / (4.0 * dr.pi)

    # Returns the probability of a random walking reaching the boundary of the ball
    @dr.syntax
    def potential(self) -> Float:
        return 4.0 * dr.pi * self.poissonKernel()

    # Implementation for volume sampling
    @dr.syntax
    def sample(self, v: Array3, sampler: PCG32):
        p, r, pdf = self._sample(v, sampler)
        return p

    @dr.syntax
    def sample_r_pdf(self, v: Array3, sampler: PCG32):
        p, r, pdf = self._sample(v, sampler)
        return p, r, pdf

    @dr.syntax
    def _sample(self, v: Array3, sampler: PCG32):
        # Calculate appropriate bound for rejection sampling
        bound = Float(1.5) / self.R
        if self.R <= self.lambda_val:
            bound = dr.maximum(dr.maximum(2.0/self.R, 2.0/self.lambda_val),
                               dr.maximum(0.5*dr.sqrt(self.R), 0.5*self.sqrtLambda))
        else:
            bound = dr.maximum(dr.minimum(2.0/self.R, 2.0/self.lambda_val),
                               dr.minimum(0.5*dr.sqrt(self.R), 0.5*self.sqrtLambda))

        # Rejection sampling
        iter = Int(0)
        active = Bool(True)
        r = Float(0.)
        pdf = Float(0.)

        while (iter < 1000) & active:
            u = sampler.next_float32()
            r = sampler.next_float32() * self.R
            pdf = self.evaluate(r) / self.norm()
            pdf_radius = pdf / (1. / (4. * dr.pi * r * r))
            iter += 1

            if u < pdf_radius / bound:
                active = Bool(False)

        r = dr.maximum(r, 1e-6)
        if r > self.R:
            r = self.R / 2.0

        return self.c + r * v, r, pdf
