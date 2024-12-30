"""
Custom distribution useful for setting up seeds.
"""
import numpy as np
class NormalDistribution:
    def __init__(self, p, var, seed=None):
        """
        Initialize a Bernoulli distribution with probability p.

        Args:
        - p (float): The probability of success (1).
        - seed (int, optional): A seed for the random number generator.
        """
        self.p = p
        self.var = var
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample(self):
        """
        Draw a sample from the Bernoulli distribution.

        Returns:
        - int: A sample from the Bernoulli distribution (0 or 1).
        """
        return self.rng.normal(loc = self.p, scale = self.var)

    def reset(self):
        """
        Reset the random number generator to ensure reproducibility.
        """
        self.rng = np.random.default_rng(self.seed)