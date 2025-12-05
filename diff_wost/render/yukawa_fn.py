from diff_wost.core.fwd import *
from diff_wost.core.math import uniform_angle
from diff_wost.render.bessel import bessk0, bessi0, bessk1, bessi1


class YukawaGreensFnBall:
    c: Array2 = Array2(0.0, 0.0)
    R: Float = Float(1.0)
    """Implementation of Yukawa potential Green's function for a ball domain"""
    lambda_val: Float = Float(1.0)

    def __init__(self, c: Array2, R: float, lambda_val: float = 1.0):
        self.c = c
        self.R = R
        self.lambda_val = lambda_val
        self.sqrtLambda = dr.sqrt(self.lambda_val)
        self.muR = self.R * self.sqrtLambda
        self.K0muR = bessk0(self.muR)
        self.I0muR = bessi0(self.muR)
        self.K1muR = bessk1(self.muR)
        self.I1muR = bessi1(self.muR)

    # Evaluates the Green's function
    @dr.syntax
    def evaluate(self, r: Float) -> Float:
        mur = r * self.sqrtLambda
        K0mur = bessk0(mur)
        I0mur = bessi0(mur)
        return (K0mur - I0mur * self.K0muR / self.I0muR) / (2.0 * dr.pi)

    @dr.syntax
    def G(self, x: Array2, y: Array2) -> Float:
        r1 = dr.maximum(1e-4, dr.norm(y - x))
        r2 = (self.R*self.R - dr.dot(x - self.c, y - self.c))/self.R
        mur1 = r1*self.sqrtLambda
        mur2 = r2*self.sqrtLambda
        K0mur1 = bessk0(mur1)
        K0mur2 = bessk0(mur2)
        I0mur1 = bessi0(mur1)
        I0mur2 = bessi0(mur2)
        Q1 = K0mur1 - I0mur1*self.K0muR/self.I0muR
        Q2 = K0mur2 - I0mur2*self.K0muR/self.I0muR
        return -(Q1 - Q2)/(2.0*dr.pi)

    # Evaluates the norm of the Green's function
    @dr.syntax
    def norm(self) -> Float:
        return (1.0 - 2.0 * dr.pi * self.poissonKernel()) / self.lambda_val

    # Evaluates the gradient norm of the Green's function
    @dr.syntax
    def gradient_norm(self, r: Float) -> Float:
        mur = r * self.sqrtLambda
        K1mur = bessk1(mur)
        I1mur = bessi1(mur)
        Qr = self.sqrtLambda * (K1mur - I1mur * self.K1muR / self.I1muR)

        return Qr / (2.0 * dr.pi * r)

    # Evaluates the gradient of the Green's function
    @dr.syntax
    def gradient(self, r: Float, y: Array2) -> Array2:
        d = y - self.c
        return d * self.gradient_norm(r)

    # Samples a point on the surface of the ball
    @dr.syntax
    def sample_surface(self, sampler: PCG32) -> tuple:
        # Generate a random direction on unit sphere and scale by radius
        theta = uniform_angle(sampler)
        dir = Array2(dr.cos(theta), dr.sin(theta))
        y = self.c + self.R * dir
        pdf = 1.0 / (2.0 * dr.pi)
        return y, pdf

    # Evaluates the Poisson Kernel (normal derivative of the Green's function)
    @dr.syntax
    def poissonKernel(self) -> Float:
        return 1.0 / (2.0 * dr.pi * self.I0muR)

    # Evaluates the radial dampening factor associated with the centered Poisson Kernel
    @dr.syntax
    def poissonKernelDampeningFactor(self, r: Float) -> Float:
        mur = r * self.sqrtLambda
        K1mur = bessk1(mur)
        I1mur = bessi1(mur)
        Q = K1mur + I1mur * self.K0muR / self.I0muR
        return mur * Q

    # Evaluates the centered Poisson Kernel at a point y with normal n
    def P(self, y: Array2, n: Array2) -> Float:
        xy = y - self.c
        xyNorm = dr.norm(xy)
        r = dr.maximum(1e-4, xyNorm)
        xy = xy / xyNorm
        return self.poissonKernelDampeningFactor(r) * dr.dot(n, xy) / (2.0 * dr.pi * r)

    # Directly evaluates the centered Poisson Kernel at a point y over the direction sampling pdf
    @dr.syntax
    def directionSampledPoissonKernel(self, y: Array2) -> Float:
        r = dr.maximum(1e-4, dr.norm(y - self.c))
        return self.poissonKernelDampeningFactor(r)

    # Evaluates the gradient of the Poisson Kernel
    @dr.syntax
    def poissonKernelGradient(self, y: Array2) -> Array2:
        d = y - self.c
        QR = self.sqrtLambda / (self.R * self.I1muR)
        return d * QR / (2.0 * dr.pi)

    # Returns the probability of a random walking reaching the boundary of the ball
    @dr.syntax
    def potential(self) -> Float:
        return 2.0 * dr.pi * self.poissonKernel()

    # Implementation for volume sampling
    @dr.syntax
    def sample(self, v: Array2, sampler: PCG32):
        p, r, pdf = self._sample(v, sampler)
        return p

    @dr.syntax
    def sample_r_pdf(self, v: Array2, sampler: PCG32):
        p, r, pdf = self._sample(v, sampler)
        return p, r, pdf

    @dr.syntax
    def _sample(self, v: Array2, sampler: PCG32):
        # Calculate appropriate bound for rejection sampling
        bound = Float(1.5) / self.R
        if self.R <= self.lambda_val:
            bound = dr.maximum(dr.maximum(2.2/self.R, 2.2/self.lambda_val),
                               dr.maximum(0.6*dr.sqrt(self.R), 0.6*self.sqrtLambda))
        else:
            bound = dr.maximum(dr.minimum(2.2/self.R, 2.2/self.lambda_val),
                               dr.minimum(0.6*dr.sqrt(self.R), 0.6*self.sqrtLambda))

        # Rejection sampling
        iter = Int(0)
        active = Bool(True)
        r = Float(0.)
        pdf = Float(0.)

        while (iter < 1000) & active:
            u = sampler.next_float32()
            r = sampler.next_float32() * self.R
            pdf = self.evaluate(r) / self.norm()
            pdf_radius = pdf / (1. / (2. * dr.pi * r))
            iter += 1

            if u < pdf_radius / bound:
                active = Bool(False)

        r = dr.maximum(r, 1e-6)
        if r > self.R:
            r = self.R / 2.0

        return self.c + r * v, r, pdf
