import numpy as np
from matching.cetralised_platforms.pac.cetralised_platform import CentralisedPlatform

from matching.matching_algo.coloring.minimum_edge_coloring import get_matchings_edge_coloring
from setup.utils import save_json


class Base_Elimination(CentralisedPlatform):
    def __init__(self,
                 delta,
                 player_preferences,
                 arm_preferences,
                 num_players,
                 mean_rewards_players,
                 variance,
                 seed,
                 random_state=None,
                 **kwargs):
        CentralisedPlatform.__init__(self,
                                     delta=delta,
                                     player_preferences=player_preferences,
                                     arm_preferences=arm_preferences,
                                     num_players=num_players,
                                     mean_rewards=mean_rewards_players,
                                     variance=variance,
                                     seed=seed,
                                     random_state=random_state)
        self.confidence_intervals = np.ones((self.num_players, self.num_players, 2)) * np.inf
        self.confidence_intervals[:, :, 0] = - np.inf
        self.alpha = np.zeros((self.num_players, self.num_players, 2))

    @staticmethod
    def is_overalping(interval_1, intraval_2):
        lb_1, ub_1 = interval_1
        lb_2, ub_2 = intraval_2
        if (ub_1 < lb_2) or (lb_1 > ub_2):
            return False
        else:
            return True

    def eliminate_agents(self, available_arms_matrix, confidence_intervals):
        available_arms = []
        for player in range(self.num_players):
            tmp_arms = []
            for arm in range(self.num_players):
                if available_arms_matrix[player, arm] == 0:
                    tmp_arms += [arm]
            available_arms += [tmp_arms]

        for player in range(self.num_players):
            for arm in available_arms[player]:
                eliminate = True
                for other_arms in set(available_arms[player]) - {arm}:
                    if self.is_overalping(confidence_intervals[player, arm, :],
                                          confidence_intervals[player, other_arms, :]):
                        eliminate = False
                        break

                if eliminate:
                    # print("ELIMINATED at round!, :", self.round)
                    # print(player, arm)
                    # print(self.confidence_intervals)
                    available_arms_matrix[player, arm] = 1

        return available_arms_matrix

    def update_confidence_interval(self, t):
        self.alpha = np.sqrt(np.log((4 * (self.num_players ** 2) * (t ** 2)) / self.delta) / (2 * t))
        self.confidence_intervals[:, :, 0] = self.avg_players_reward - self.alpha
        self.confidence_intervals[:, :, 1] = self.avg_players_reward + self.alpha

    def update_reward(self, reward, num_plays, old_rewards):
        new_rewards = ((old_rewards * num_plays) + reward) / (num_plays + 1)
        return new_rewards

    def get_matches(self, available_arms):
        availability_matrix = np.copy(available_arms)
        matchings = get_matchings_edge_coloring(num_players=self.num_players,
                                                adjacency_matrix=1 - availability_matrix)
        return matchings, len(matchings)

    def stopping_rule(self, **kwargs):
        raise NotImplementedError("define stopping_rule")

    def run(self):
        # exploration steps
        available_arms = np.zeros((self.num_players, self.num_players))  # 0 for available, 1 for eliminated

        flag = True
        self.round = 0
        self.matching_rounds = 0
        self.num_samples = 0
        self.optimal_stable = []
        self.optimal_stable2 = []
        self.pref_true = []
        self.pref_up_to_stable_match = []
        self.optimal_match_round_id = []
        self.optimal_match_matching_id = []
        self.optimal_match_sample_id = []
        fist_matching_elim = None
        while flag:
            # generate matchings
            matches, tmp_matching_rounds = self.get_matches(np.copy(available_arms))
            # keep track of the first round we sample less matching - eliminate a matching
            if (len(matches) < self.num_players) and fist_matching_elim is None:
                fist_matching_elim = self.round + 1

            self.round += 1
            for tmp_match in matches:
                self.matching_rounds += 1
                # sample rewards and update empirical estimates
                for (player, arm) in tmp_match:
                    tuple_match = (player, arm)
                    tmp_player_reward = self.sample_reward(player=player,
                                                           arm=arm,
                                                           mean_rewards=self.mean_rewards)
                    # update
                    self.avg_players_reward[tuple_match] = self.update_reward(reward=tmp_player_reward,
                                                                              num_plays=self.num_plays[tuple_match],
                                                                              old_rewards=self.avg_players_reward[
                                                                                  tuple_match])
                    self.num_samples += 1
                    self.num_plays[tuple_match] += 1
                self.calc_pac_metrics()  # this is for the anytime performance of the algorithms

            stable_flag = 0
            self.stability_over_time = np.append(self.stability_over_time, stable_flag)
            player_preferences = self.preferences_from_rewards(self.avg_players_reward)
            pref_flag = player_preferences == self.player_preferences
            self.pref_over_time = np.append(self.pref_over_time, pref_flag)
            # 2. update confidence intervals
            self.update_confidence_interval(t=self.round)

            # 3. eliminate pairs
            available_arms = self.eliminate_agents(available_arms_matrix=available_arms,
                                                   confidence_intervals=self.confidence_intervals)
            # 4. Stopping rule
            flag = self.stopping_rule(available_arms=available_arms)

        # get final metrics
        player_preferences = self.preferences_from_rewards(self.avg_players_reward)
        final_match = self.match(player_preferences, self.arm_preferences)
        stable_flag = self.is_stable(final_match.tolist())
        pref_flag = player_preferences == self.player_preferences

        self.optimal_stable = np.all(final_match == self.optimal_match)
        self.sample_complexity = self.matching_rounds + 1
        self.stability = stable_flag
        self.pref_over_time = np.append(self.pref_over_time, pref_flag)
        try:
            self.max_pref = np.where(self.pref_over_time == 0)[0].max()
        except:
            self.max_pref = 0
        return

    def calc_pac_metrics(self):
        # 5. check stability adn preferences
        # A. check if preferences are correct
        tmp_player_preferences = self.preferences_from_rewards(self.avg_players_reward)
        tmp_gs = self.match(tmp_player_preferences, self.arm_preferences)

        # B.  check if preferences up to the stable match are are correct
        self.pref_true += [tmp_player_preferences == self.player_preferences]
        res = []
        for p in range(self.num_players):
            # 1. get preference up to stable match
            player_stable_match = self.optimal_match[p]
            index_true_pref = self.player_preferences[p].index(player_stable_match)
            tmp_true_pref = self.player_preferences[p][:index_true_pref + 1]

            index_pref = tmp_player_preferences[p].index(player_stable_match)
            tmp_est_pref = self.player_preferences[p][:index_pref + 1]
            # 2. check if is correct
            res += [tmp_true_pref == tmp_est_pref]
        correct_preference_up_stable_match = np.all(res)
        self.pref_up_to_stable_match += [int(correct_preference_up_stable_match) * 1]

        # C. match is optimal stable
        self.optimal_stable2 += [tmp_gs.tolist() == self.optimal_match.tolist()]
        self.optimal_match_round_id += [self.round]
        self.optimal_match_matching_id += [self.matching_rounds]
        self.optimal_match_sample_id += [self.num_samples]

    def save_results(self, save_path):
        res = {"pref_over_time": self.pref_true,
               "pref_up_to_stable_match_over_time": self.pref_up_to_stable_match,
               "stability_over_time": self.optimal_stable2,
               "round_index": self.optimal_match_round_id,
               "matching_index": self.optimal_match_matching_id,
               "sample_index": self.optimal_match_sample_id}
        save_json(res, save_path + '/check_results.json')


class NaiveUniformlySampling_pac(Base_Elimination):
    """
    This is similar to ETC with unknown Delta.
    It samples uniformly every arm until we eliminate all arms.
    """

    def __init__(self,
                 delta,
                 player_preferences,
                 arm_preferences,
                 num_players,
                 mean_rewards_players,
                 variance,
                 random_state=None,
                 **kwargs):
        Base_Elimination.__init__(self,
                                  delta=delta,
                                  player_preferences=player_preferences,
                                  arm_preferences=arm_preferences,
                                  num_players=num_players,
                                  mean_rewards_players=mean_rewards_players,
                                  variance=variance,
                                  random_state=random_state,
                                  **kwargs)

    def get_matches(self, available_arms):
        """
        Round-robin
        """
        matchings = []
        for h in range(self.num_players):
            t = self.round + h
            matching = [(player, (t + player) % self.num_players) for player in range(self.num_players)]
            matchings += [np.array(matching)]
        return matchings, len(matchings)

    def stopping_rule(self, available_arms, **kwargs):
        flag = True
        if np.sum(available_arms) == self.num_players * self.num_players:
            flag = False
        return flag


class Elimination_Algo_pac(Base_Elimination):
    def __init__(self,
                 delta,
                 player_preferences,
                 arm_preferences,
                 num_players,
                 mean_rewards_players,
                 variance,
                 random_state=None,
                 **kwargs):
        Base_Elimination.__init__(self,
                                  delta=delta,
                                  player_preferences=player_preferences,
                                  arm_preferences=arm_preferences,
                                  num_players=num_players,
                                  mean_rewards_players=mean_rewards_players,
                                  variance=variance,
                                  random_state=random_state,
                                  **kwargs)

    def stopping_rule(self, available_arms, **kwargs):
        flag = True
        if np.sum(available_arms) == self.num_players * self.num_players:
            flag = False
        return flag


class ImproveElimination_Algo_pac(Base_Elimination):
    def __init__(self,
                 delta,
                 player_preferences,
                 arm_preferences,
                 num_players,
                 mean_rewards_players,
                 variance,
                 random_state=None,
                 **kwargs):
        Base_Elimination.__init__(self,
                                  delta=delta,
                                  player_preferences=player_preferences,
                                  arm_preferences=arm_preferences,
                                  num_players=num_players,
                                  mean_rewards_players=mean_rewards_players,
                                  variance=variance,
                                  random_state=random_state,
                                  **kwargs)

    def stopping_rule(self, available_arms, **kwargs):
        # gale shapley algo
        player_preferences = self.preferences_from_rewards(self.avg_players_reward)
        gs_match = self.match(player_preferences, self.arm_preferences)

        flag = False  # flag = False -> we stop
        for player in range(self.num_players):
            ms_p = gs_match[player]
            index_pref = player_preferences[player].index(ms_p)
            arms_to_check = player_preferences[player][:index_pref + 1]
            for arm in arms_to_check:
                if available_arms[player, arm] == 0:  # arms is not eliminated so we have to explore more
                    flag = True  # flag= True -> we continue
                    break
            if flag:
                break

        if not flag:
            print("stop", available_arms)
        return flag


class AdaptiveImproveElimination_Algo_pac(Base_Elimination):
    def __init__(self,
                 delta,
                 player_preferences,
                 arm_preferences,
                 num_players,
                 mean_rewards_players,
                 variance,
                 random_state=None,
                 **kwargs):
        Base_Elimination.__init__(self,
                                  delta=delta,
                                  player_preferences=player_preferences,
                                  arm_preferences=arm_preferences,
                                  num_players=num_players,
                                  mean_rewards_players=mean_rewards_players,
                                  variance=variance,
                                  random_state=random_state,
                                  **kwargs)

    def stopping_rule(self, available_arms, **kwargs):
        flag = True
        if np.sum(available_arms) == self.num_players * self.num_players:
            flag = False
        return flag

    def eliminate_agents(self, available_arms_matrix, confidence_intervals):
        # available_arms_matrix not used
        # 1. get gale shapley match
        player_preferences = self.preferences_from_rewards(self.avg_players_reward)
        gs_match = self.match(player_preferences, self.arm_preferences)

        # 2. get arms to explore
        active_arms = np.ones((self.num_players, self.num_players))  # one indicate eliminated arms
        for player in range(self.num_players):
            ms_p = gs_match[player]
            index_pref = player_preferences[player].index(ms_p)
            arms_to_check = player_preferences[player][:index_pref + 1]
            for arm in arms_to_check:
                for other_arm in set(range(self.num_players)) - {arm}:
                    if self.is_overalping(confidence_intervals[player, arm, :],
                                          confidence_intervals[player, other_arm, :]):
                        active_arms[player, arm] = 0
                        active_arms[player, other_arm] = 0

        return active_arms

    def update_confidence_interval(self, t):
        self.alpha = np.sqrt(
            np.log((4 * (self.num_players ** 2) * (self.num_plays ** 2)) / self.delta) / (2 * self.num_plays))
        self.confidence_intervals[:, :, 0] = self.avg_players_reward - self.alpha
        self.confidence_intervals[:, :, 1] = self.avg_players_reward + self.alpha
