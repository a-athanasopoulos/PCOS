"""
Base class for the algorithms
"""
import numpy as np
from matching.matching_algo.is_stable import is_stable
class BasePlatform(object):
    """
    Base class for cetralised platforms for gaussian rewards
    """

    @staticmethod
    def preferences_from_rewards(rewards):
        return np.argsort(-rewards).tolist()  # - for arg sort in descending order

    def is_stable(self, match):
        stable = is_stable(match,
                           self.arm_preferences,
                           self.player_preferences,
                           self.num_players)
        return stable

    def match(self, **kwargs):
        """
        performing Gale Shapley algorithm for matching
        """
        """ Sample reward for given player, arm"""
        raise NotImplementedError("Please implement your platform method")

    def sample_reward(self, **kwargs):
        """ Sample reward for given player, arm"""
        raise NotImplementedError("Please implement your platform method")

    def update_reward(self, **kwargs):
        """ Update reward for given player, arm """
        raise NotImplementedError("Please implement your platform method")

    def run(self):
        """ Implement the platform algorithm for the matching"""
        raise NotImplementedError("Please implement your platform method")
