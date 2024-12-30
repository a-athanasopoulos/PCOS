import math

import numpy as np

from matching.cetralised_platforms.pac.cetralised_platform import CentralisedPlatform


class ETC_pac(CentralisedPlatform):
    def __init__(self,
                 delta,
                 Delta_min,
                 player_preferences,
                 arm_preferences,
                 num_players,
                 mean_rewards_players,
                 variance,
                 random_state=None,
                 **kwargs):
        CentralisedPlatform.__init__(self,
                                     delta=delta,
                                     player_preferences=player_preferences,
                                     arm_preferences=arm_preferences,
                                     num_players=num_players,
                                     mean_rewards=mean_rewards_players,
                                     variance=variance,
                                     random_state=random_state)

        self.N = self.num_players
        self.K = self.num_players
        self.h = 4 * np.log(self.K * self.N / delta) / (Delta_min ** 2)
        self.sample_complexity = math.ceil(self.h * self.num_players)
        print("Sample complexity: ", self.sample_complexity)
        print(f"Variables: \n ")
        print(f"N: {self.N}")
        print(f"K: {self.K}")
        print(f"δ: {delta}")
        print(f"Δ: {Delta_min}")

    def run(self):
        # exploration steps
        for t in range(self.sample_complexity):
            tmp_match = [(t + player) % self.num_players for player in range(self.num_players)]
            # update metrics
            stable_flag = self.is_stable(tmp_match)
            self.stability_over_time = np.append(self.stability_over_time, stable_flag)
            player_preferences = self.preferences_from_rewards(self.avg_players_reward)
            pref_flag = player_preferences == self.player_preferences
            self.pref_over_time = np.append(self.pref_over_time, pref_flag)

            optimal_regret = np.zeros(self.num_players)
            pessimal_regret = np.zeros(self.num_players)
            for player, arm in enumerate(tmp_match):
                tuple_match = (player, arm)
                tmp_player_reward = self.sample_reward(player=player, arm=arm, mean_rewards=self.mean_rewards)
                self.num_plays[tuple_match] += 1
                self.avg_players_reward[tuple_match] = self.update_reward(reward=tmp_player_reward,
                                                                          num_plays=self.num_plays[tuple_match],
                                                                          old_rewards=self.avg_players_reward[tuple_match])
                # update


                # update regret
                optimal_regret[player] = self.mean_rewards[player, self.optimal_match[player]] - self.mean_rewards[
                    tuple_match]
                pessimal_regret[player] = self.mean_rewards[player, self.pessimal_match[player]] - self.mean_rewards[
                    tuple_match]
            self.player_optimal_regret += [optimal_regret]
            self.player_pessimal_regret += [pessimal_regret]

        # exploitation steps
        player_preferences = self.preferences_from_rewards(self.avg_players_reward)
        final_match = self.match(player_preferences, self.arm_preferences)
        stable_flag = self.is_stable(final_match.tolist())
        pref_flag = player_preferences == self.player_preferences

        self.optimal_stable = np.all(final_match == self.optimal_match)
        self.sample_complexity = self.sample_complexity
        self.stability = stable_flag
        self.pref_over_time = np.append(self.pref_over_time, pref_flag)
        self.max_pref = np.where(self.pref_over_time == 0)[0].max()
        return
