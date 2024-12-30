"""
Base class for centralised algorithms
"""
import numpy as np

from matching.cetralised_platforms.pac.BasePlatform import BasePlatform
from matching.distributions.bernoulli import BernoulliDistribution
from matching.utils import inv_matching
from matching.matching_algo.gale_shapley import gale_shapley_algo

class CentralisedPlatform(BasePlatform):
    """
    Base class for centralised platforms for gaussian rewards
    """

    def __init__(self,
                 delta,
                 player_preferences,
                 arm_preferences,
                 num_players,
                 mean_rewards,
                 variance,
                 seed,
                 random_state=None):
        self.reward_distributions = []
        self.seed = seed
        # global variables
        self.delta = delta
        self.num_players = num_players
        self.variance = variance
        self.num_plays = np.zeros((self.num_players, self.num_players))
        self.random_state = random_state

        #
        self.player_preferences = player_preferences
        self.arm_preferences = arm_preferences
        self.mean_rewards = np.array(mean_rewards)
        self.avg_players_reward = np.zeros((self.num_players, self.num_players))

        # metrics
        self.optimal_match = self.match(self.player_preferences, self.arm_preferences)  # optimal mode 0
        self.pessimal_match = self.match(self.player_preferences, self.arm_preferences, mode=1)  # pessimal mode 1

        # additional metrics
        self.player_pessimal_regret = []
        self.player_optimal_regret = []
        self.stability_over_time = np.array([])
        self.pref_over_time = np.array([])

        self.optimal_stable = 0
        self.stability = 0
        self.max_pref = 0
        self.sample_complexity = 0
        self.set_rewards_distributions(seed=seed,
                                       num_players=num_players,
                                       num_arms=num_players,
                                       mean_rewards=self.mean_rewards)  # this init the reward distribution
        self.collected_samples = []
        for p in range(self.num_players):
            self.collected_samples += [[]]
            for a in range(self.num_players):
                self.collected_samples[p] += [[]]

    def match(self, player_preferences, arm_preferences, mode=0, **kwargs):
        """
        performing Gale Shapley algorithm for matching
        """
        if mode == 0:  # mode for optimal stable match otherwise pessimal matching
            gale_shapley_match = gale_shapley_algo(player_ranking=player_preferences,
                                                   arms_rankings=arm_preferences,
                                                   num_players=self.num_players,
                                                   num_arms=self.num_players)
        else:  # this is the pessimal matching
            gale_shapley_match = gale_shapley_algo(player_ranking=arm_preferences,
                                                   arms_rankings=player_preferences,
                                                   num_players=self.num_players,
                                                   num_arms=self.num_players)
            gale_shapley_match = inv_matching(gale_shapley_match).astype("int64")

        return gale_shapley_match

    def set_rewards_distributions(self, seed, num_players, num_arms, mean_rewards):
        np.random.seed(seed=seed)
        sub_seeds = []
        for player in range(num_players):
            reward_distributions = []
            for arm in range(num_arms):
                sub_seed = np.random.randint(10000)
                sub_seeds += [sub_seed]
                reward_distributions += [BernoulliDistribution(p=min(1, mean_rewards[player][arm]),
                                                               seed=sub_seed)]
            self.reward_distributions += [reward_distributions]
        print(sub_seeds)

    def sample_reward(self, player, arm, mean_rewards):
        sample = self.reward_distributions[player][arm].sample()
        self.collected_samples[player][arm] += [sample]
        return sample

    def run(self):
        """ Implement the platform algorithm for the matching"""
        raise NotImplementedError("Please implement your platform method")
