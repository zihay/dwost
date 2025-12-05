"""Green's function implementations for 2D Walk on Spheres.

This module provides Green's function representations for balls/circles
used in the Walk on Stars method for solving Poisson equations.
"""

from dataclasses import dataclass

import numpy as np

from diff_wost.core.fwd import PCG32, Array2, Bool, Float, Int, dr


@dataclass
class GreensBall:
    """Green's function for a 2D ball (circle).

    This class implements the Green's function G(x, y) for the Laplacian
    on a ball, along with its derivatives and sampling methods needed for
    Walk on Spheres Monte Carlo integration.

    Attributes:
        c: Center of the ball.
        R: Radius of the ball.
    """

    c: Array2 = Array2(0.0, 0.0)
    R: float = 1.0

    def G(self, x: Array2, y: Array2) -> Float:
        """Evaluate the Green's function G(x, y).

        The Green's function satisfies: ΔG(x, y) = δ(x - y) with G = 0 on boundary.

        Args:
            x: First point (typically the source/pole).
            y: Second point (typically the evaluation point).

        Returns:
            Value of the Green's function at the given points.
        """
        x = x - self.c
        y = y - self.c
        x = x / self.R
        y = y / self.R
        v = dr.norm(x - y)
        return -1.0 / (2 * np.pi) * dr.log((1.0 - dr.dot(x, y)) / v)

    def dGdx(self, x: Array2, y: Array2) -> Array2:
        """Compute the gradient of G with respect to x.

        Args:
            x: First point.
            y: Second point.

        Returns:
            Gradient ∇_x G(x, y).
        """
        v = y - x
        r = dr.norm(v)
        a = v / (r * r)
        b = y / (self.R * self.R - dr.dot(x, y))
        return -1.0 / (2.0 * np.pi) * (a - b)

    def dGdy(self, x: Array2, y: Array2) -> Array2:
        """Compute the gradient of G with respect to y.

        Args:
            x: First point.
            y: Second point.

        Returns:
            Gradient ∇_y G(x, y).
        """
        x = x - self.c
        y = y - self.c
        return -1.0 / (2 * np.pi) * (-y) / dr.squared_norm(y)

    def P(self, x: Array2, y: Array2, ny: Array2) -> Float:
        """Compute the Poisson kernel P(x, y; n_y).

        The Poisson kernel is the normal derivative of the Green's function:
        P(x, y) = ∂G/∂n_y

        Args:
            x: Interior point.
            y: Boundary point.
            ny: Outward normal at y.

        Returns:
            Value of the Poisson kernel.
        """
        x = x - self.c
        y = y - self.c
        x = x / self.R
        y = y / self.R
        v = x - y
        a = v / dr.squared_norm(v)
        b = x / (1 - dr.dot(x, y))
        return -1.0 / (2 * np.pi) * (dr.dot(a, ny) - dr.dot(b, ny)) / self.R

    def dPdnx(self, x: Array2, y: Array2, nx: Array2, ny: Array2) -> Float:
        """Compute the derivative of Poisson kernel with respect to n_x.

        Args:
            x: Interior point.
            y: Boundary point.
            nx: Normal direction at x.
            ny: Outward normal at y.

        Returns:
            Derivative ∂P/∂n_x.
        """
        r = dr.norm(x - y)
        a = 2.0 * x / (self.R * r * r)
        b = (
            2.0
            * (dr.squared_norm(x) - self.R * self.R)
            * (x - y)
            / (self.R * r * r * r * r)
        )
        return -1.0 / (2 * np.pi) * (dr.dot(a, ny) - dr.dot(b, ny))

    def dPdny(self, x: Array2, y: Array2, ny: Array2) -> Float:
        """Compute the derivative of Poisson kernel with respect to n_y.

        Args:
            x: Interior point.
            y: Boundary point.
            ny: Outward normal at y.

        Returns:
            Derivative ∂P/∂n_y.
        """
        u = x - y
        r = dr.norm(u)
        r2 = r * r
        r4 = r2 * r2
        a = -(dr.dot(ny, ny)) / r2
        b = 2.0 * dr.dot(u, ny) * dr.dot(u, ny) / r4
        c = self.R * self.R - dr.dot(x, y)
        d = dr.dot(x, ny) * dr.dot(x, ny) / (c * c)
        return -1.0 / (2.0 * np.pi) * ((a + b) - d)

    @dr.syntax
    def sample(self, v: Array2, sampler: PCG32) -> Array2:
        """Sample a point inside the ball along direction v.

        Args:
            v: Direction vector (should be unit length).
            sampler: Random number generator.

        Returns:
            Sampled point inside the ball.
        """
        p, r, pdf = self._sample(v, sampler)
        return p

    @dr.syntax
    def sample_r_pdf(self, v: Array2, sampler: PCG32) -> tuple:
        """Sample a point and return with radius and PDF.

        Args:
            v: Direction vector (should be unit length).
            sampler: Random number generator.

        Returns:
            Tuple of (sampled point, radius, PDF value).
        """
        p, r, pdf = self._sample(v, sampler)
        return p, r, pdf

    @dr.syntax
    def _sample(self, v: Array2, sampler: PCG32) -> tuple:
        """Internal sampling implementation using rejection sampling.

        Samples radius r from the PDF proportional to r * ln(R/r).

        Args:
            v: Direction vector.
            sampler: Random number generator.

        Returns:
            Tuple of (sampled point, radius, PDF value).
        """
        bound = Float(1.5) / self.R
        iteration = Int(0)
        active = Bool(True)
        r = Float(0.0)
        pdf = Float(0.0)

        while (iteration < 1000) & active:
            u = sampler.next_float32()
            r = sampler.next_float32() * self.R
            pdf = self.evaluate(r) / self.norm()
            pdf_radius = pdf / (1.0 / (2.0 * np.pi * r))
            iteration += 1

            if u < pdf_radius / bound:
                active = Bool(False)

        r = dr.maximum(r, 1e-6)
        if r > self.R:
            r = self.R / 2.0

        return self.c + r * v, r, pdf

    @dr.syntax
    def evaluate(self, r: Float) -> Float:
        """Evaluate the radial part of the Green's function.

        Args:
            r: Radius from center.

        Returns:
            Green's function value at radius r.
        """
        return dr.log(self.R / r) / (2.0 * np.pi)

    @dr.syntax
    def norm(self) -> Float:
        """Compute the normalization constant (integral of G over the ball).

        Returns:
            Normalization constant R²/4.
        """
        return self.R * self.R / 4.0

    @dr.syntax
    def gradient_norm(self, r: Float) -> Float:
        """Compute the norm of the gradient at radius r.

        Args:
            r: Radius from center.

        Returns:
            Magnitude of the gradient.
        """
        r2 = r * r
        return (1.0 / r2 - 1.0 / (self.R * self.R)) / (2.0 * np.pi)

    @dr.syntax
    def gradient(self, r: Float, y: Array2) -> Array2:
        """Compute the gradient of the Green's function.

        Args:
            r: Radius from center.
            y: Point where gradient is evaluated.

        Returns:
            Gradient vector at y.
        """
        d = y - self.c
        return d * self.gradient_norm(r)

    def directionSampledPoissonKernel(self, y: Array2) -> Float:
        """Compute direction-sampled Poisson kernel weight.

        Args:
            y: Boundary point.

        Returns:
            Weight factor (always 1.0 for uniform direction sampling).
        """
        return Float(1.0)

    def poissonKernel(self) -> Float:
        """Compute the uniform Poisson kernel value.

        Returns:
            Poisson kernel value 1/(2π).
        """
        return 1.0 / (2.0 * np.pi)

    def poissonKernelGradient(self, y: Array2) -> Array2:
        """Compute the gradient of the Poisson kernel.

        Used for gradient estimation in derivative Walk on Stars.

        Args:
            y: Point where gradient is evaluated.

        Returns:
            Gradient of the Poisson kernel.
        """
        d = y - self.c
        return 2.0 * d / (2.0 * np.pi * self.R * self.R)
