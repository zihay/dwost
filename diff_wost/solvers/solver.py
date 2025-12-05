"""Base solver classes for Walk on Stars Monte Carlo methods.

This module provides the base `Solver` class and `CustomSolver` for implementing
PDE solvers using Walk on Spheres/Stars methods with Dr.Jit for GPU acceleration.
"""

from dataclasses import dataclass
from typing import Any

from diff_wost.core.fwd import PCG32, Array2, Float, Int, dr
from diff_wost.core.math import sample_tea_32


@dataclass
class Solver:
    """Base class for Walk on Stars Monte Carlo PDE solvers.

    This class provides the infrastructure for Monte Carlo estimation of PDE solutions
    using the Walk on Spheres/Stars method. Subclasses should implement the `u` method
    to define the specific walking strategy.

    Attributes:
        eps: Epsilon threshold for termination near Dirichlet boundary.
        r_min: Minimum ball radius to prevent numerical instability.
        nwalks: Number of independent walks per evaluation point (wavefront size).
        nsubwalks: Number of sequential subwalks per walk (reduces variance).
        nsteps: Maximum number of steps per walk before termination.
    """

    eps: float = 1e-3
    r_min: float = 1e-4
    nwalks: int = 1000
    nsubwalks: int = 1
    nsteps: int = 128

    def solve(self, p: Array2, scene: Any, seed: int = 0) -> Float:
        """Solve the PDE at given evaluation points.

        Uses Monte Carlo estimation by launching multiple random walks from each point
        and averaging the results.

        Args:
            p: Array of 2D evaluation points.
            scene: Scene object containing geometry and boundary conditions.
            seed: Random seed for reproducibility.

        Returns:
            Array of PDE solution values at each input point.
        """
        npoints = dr.width(p)
        nsamples = npoints * self.nwalks
        result = dr.zeros(Float, npoints)

        # Expand points for wavefront parallelism
        idx = dr.arange(Int, nsamples)
        p = dr.repeat(p, self.nwalks)
        if self.nwalks > 1:
            idx //= self.nwalks

        # Initialize RNG with TEA hashing for decorrelated sequences
        v0, v1 = sample_tea_32(seed, idx)
        dr.eval(v0, v1)
        sampler = PCG32(size=nsamples, initstate=v0, initseq=v1)

        # Perform walks and accumulate results
        value = self.walk(p, scene, sampler)
        dr.scatter_reduce(dr.ReduceOp.Add, result, value, idx)

        if self.nwalks > 1:
            result /= self.nwalks
        return result

    @dr.syntax
    def walk(self, p: Array2, scene: Any, sampler: PCG32) -> Float:
        """Execute random walks from given starting points.

        Performs multiple subwalks and averages the results for variance reduction.

        Args:
            p: Starting points for the walks.
            scene: Scene object.
            sampler: Random number generator.

        Returns:
            Estimated PDE solution values.
        """
        result = Float(0.0)
        i = Int(0)
        while i < self.nsubwalks:
            result += self.u(p, scene, sampler)
            i += 1
        return result / self.nsubwalks

    @dr.syntax
    def u(self, p: Array2, scene: Any, sampler: PCG32) -> Float:
        """Compute the PDE solution using a single random walk.

        This method should be overridden by subclasses to implement
        specific walking strategies (WoS, WoSt, etc.).

        Args:
            p: Starting point for the walk.
            scene: Scene object.
            sampler: Random number generator.

        Returns:
            Estimated solution value.

        Raises:
            NotImplementedError: This base method must be overridden.
        """
        raise NotImplementedError("Subclasses must implement the 'u' method")

    def solve_dude(self, p: Array2, e: Array2, scene: Any, seed: int = 0) -> Float:
        """Solve for directional derivatives using DUDE (Direction-Uniform Derivative Estimator).

        Args:
            p: Evaluation points.
            e: Direction vectors for derivative estimation.
            scene: Scene object.
            seed: Random seed.

        Returns:
            Directional derivative estimates at each point.
        """
        npoints = dr.width(p)
        nsamples = npoints * self.nwalks
        result = dr.zeros(Float, npoints)

        idx = dr.arange(Int, nsamples)
        p = dr.repeat(p, self.nwalks)
        e = dr.repeat(e, self.nwalks)
        if self.nwalks > 1:
            idx //= self.nwalks

        v0, v1 = sample_tea_32(seed, idx)
        dr.eval(v0, v1)
        sampler = PCG32(size=nsamples, initstate=v0, initseq=v1)

        value = self.walk_dude(p, e, scene, sampler)
        dr.scatter_reduce(dr.ReduceOp.Add, result, value, idx)

        if self.nwalks > 1:
            result /= self.nwalks
        return result

    @dr.syntax
    def walk_dude(self, p: Array2, e: Array2, scene: Any, sampler: PCG32) -> Float:
        """Execute derivative estimation walks.

        Args:
            p: Starting points.
            e: Direction vectors.
            scene: Scene object.
            sampler: Random number generator.

        Returns:
            Derivative estimates.
        """
        result = Float(0.0)
        i = Int(0)
        while i < self.nsubwalks:
            result += self.dude(p, e, scene, sampler)
            i += 1
        return result / self.nsubwalks

    @dr.syntax
    def dude(self, p: Array2, e: Array2, scene: Any, sampler: PCG32) -> Float:
        """Compute directional derivative using a single walk.

        Args:
            p: Starting point.
            e: Direction vector.
            scene: Scene object.
            sampler: Random number generator.

        Returns:
            Derivative estimate.

        Raises:
            NotImplementedError: This base method must be overridden.
        """
        raise NotImplementedError("Subclasses must implement the 'dude' method")


class Attached:
    """Wrapper to prevent Dr.Jit from detaching values during custom operations.

    This is used to preserve AD graph connections when passing scene data
    through custom differentiable operations.
    """

    def __init__(self, value: Any) -> None:
        """Initialize with the value to protect.

        Args:
            value: The value to keep attached to the AD graph.
        """
        self.value = value


class _CustomSolver(dr.CustomOp):
    """Custom differentiable operation for PDE solving.

    This enables automatic differentiation through the Monte Carlo solver
    by defining custom forward and backward passes.
    """

    def eval(
        self,
        fwd_solver: Solver,
        bwd_solver: Solver,
        p: Array2,
        scene: Any,
        attached_scene: Attached,
        seed: int = 0,
        dummy: Float = Float(0.0),
    ) -> Float:
        """Evaluate the forward pass (primal computation).

        Args:
            fwd_solver: Solver for forward evaluation.
            bwd_solver: Solver for backward (gradient) evaluation.
            p: Evaluation points.
            scene: Scene object (detached version).
            attached_scene: Scene wrapped to preserve AD connections.
            seed: Random seed.
            dummy: Dummy variable to ensure AD graph connection.

        Returns:
            Detached solution values.
        """
        self.fwd_solver = fwd_solver
        self.bwd_solver = bwd_solver
        self.p = p
        self.scene = attached_scene.value
        self.seed = seed
        # Return detached value - gradients computed in backward()
        return dr.detach(fwd_solver.solve(p, scene, seed))

    def forward(self) -> None:
        """Forward-mode automatic differentiation."""
        value = self.fwd_solver.solve(self.p, self.scene, self.seed)
        dr.forward_to(value)
        self.set_grad_out(value)

    def backward(self) -> None:
        """Backward-mode automatic differentiation (backpropagation)."""
        value = self.bwd_solver.solve(self.p, self.scene, self.seed)
        dr.set_grad(value, self.grad_out())
        dr.enqueue(dr.ADMode.Backward, value)
        dr.traverse(dr.ADMode.Backward, dr.ADFlag.ClearInterior)


@dataclass
class CustomSolver:
    """Differentiable solver combining forward and backward solvers.

    This class wraps separate forward and backward solvers into a single
    differentiable operation that can be used with Dr.Jit's automatic
    differentiation system.

    Attributes:
        fwd_solver: Solver used for forward evaluation.
        bwd_solver: Solver used for gradient computation.
    """

    fwd_solver: Solver = None
    bwd_solver: Solver = None

    def solve(self, p: Array2, scene: Any, seed: int = 0) -> Float:
        """Solve the PDE with automatic differentiation support.

        Args:
            p: Evaluation points.
            scene: Scene object (may contain differentiable parameters).
            seed: Random seed for reproducibility.

        Returns:
            Solution values that support automatic differentiation.
        """
        dummy = Float(0.0)
        dr.enable_grad(dummy)
        return dr.custom(
            _CustomSolver,
            self.fwd_solver,
            self.bwd_solver,
            p,
            scene,
            Attached(scene),
            seed,
            dummy,
        )
