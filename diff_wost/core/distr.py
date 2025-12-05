"""Discrete probability distribution for importance sampling.

This module provides a discrete distribution class that supports efficient
sampling and PDF evaluation, used for importance sampling in Monte Carlo
integration (e.g., selecting primitives for boundary sampling).
"""

from diff_wost.core.fwd import Bool, Float, Int, dr


class DiscreteDistribution:
    """A discrete probability distribution supporting efficient sampling.

    This class maintains a probability mass function (PMF) and its cumulative
    distribution function (CDF), enabling O(log n) sampling via binary search.

    Attributes:
        pmf: Probability mass function (unnormalized weights).
        cdf: Cumulative distribution function.
        sum: Sum of all PMF values.
        normalization: Reciprocal of sum for normalization.
        valid: Valid index range [min, max] for binary search.
    """

    pmf: Float
    cdf: Float
    sum: Float
    normalization: Float
    valid: list

    def __init__(self, values: Float = Float()) -> None:
        """Initialize the distribution with given weights.

        Args:
            values: Array of unnormalized probability weights.
        """
        self.pmf = Float(values)
        self.update()

    def update(self) -> None:
        """Recompute the CDF after PMF changes."""
        self.compute_cdf()

    def compute_cdf(self) -> None:
        """Compute the cumulative distribution function from PMF."""
        self.cdf = dr.cumsum(self.pmf)
        self.valid = [0, dr.width(self.pmf) - 1]
        self.sum = dr.gather(Float, self.cdf, self.valid[1], active=True)
        self.normalization = dr.rcp(self.sum)
        dr.make_opaque(self.valid, self.sum, self.normalization)

    def sample(self, sample: Float, active: Bool = Bool(True)) -> Int:
        """Sample an index from the distribution.

        Uses binary search on the CDF for O(log n) complexity.

        Args:
            sample: Uniform random sample in [0, 1].
            active: Mask for active lanes.

        Returns:
            Sampled index.
        """
        sample = sample * self.sum

        def func(index: Int) -> Bool:
            value = dr.gather(Float, self.cdf, index, active)
            return ((value < sample) | (value == 0)) & (value != self.sum)

        return dr.binary_search(self.valid[0], self.valid[1], func)

    def sample_reuse(
        self, value: Float, active: Bool = Bool(True)
    ) -> tuple[Int, Float]:
        """Sample an index and reuse the random value.

        The random value is transformed to be reusable for subsequent
        sampling within the selected category.

        Args:
            value: Uniform random sample in [0, 1].
            active: Mask for active lanes.

        Returns:
            Tuple of (sampled index, reusable random value in [0, 1]).
        """
        index = self.sample(value, active)
        pmf = self.eval_pmf_normalized(index, active)
        cdf = self.eval_cdf_normalized(index - 1, active & (index > 0))
        return index, (value - cdf) / pmf

    def sample_reuse_pmf(
        self, value: Float, active: Bool = Bool(True)
    ) -> tuple[Int, Float, Float]:
        """Sample an index, return reused value and PMF.

        Args:
            value: Uniform random sample in [0, 1].
            active: Mask for active lanes.

        Returns:
            Tuple of (sampled index, reusable random value, normalized PMF).
        """
        index = self.sample(value, active)
        pmf = self.eval_pmf_normalized(index, active)
        cdf = self.eval_cdf_normalized(index - 1, active & (index > 0))
        return index, (value - cdf) / pmf, pmf

    def eval_pmf_normalized(self, index: Int, active: Bool = Bool(True)) -> Float:
        """Evaluate the normalized PMF at a given index.

        Args:
            index: Index to evaluate.
            active: Mask for active lanes.

        Returns:
            Normalized probability mass at the given index.
        """
        return dr.gather(Float, self.pmf, index, active) * self.normalization

    @dr.syntax
    def eval_cdf_normalized(self, index: Int, active: Bool = Bool(True)) -> Float:
        """Evaluate the normalized CDF at a given index.

        Args:
            index: Index to evaluate (value is CDF up to and including this index).
            active: Mask for active lanes.

        Returns:
            Normalized cumulative probability up to the given index.
        """
        cdf = Float(0.0)
        if active:
            cdf = dr.gather(Float, self.cdf, index, active) * self.normalization
        return cdf
