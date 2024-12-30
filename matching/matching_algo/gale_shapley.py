import numpy as np

from matching.utils import inv_matching

def gale_shapley_algo(arms_rankings, player_ranking, num_players, num_arms):
    matched = np.zeros(num_players, bool)
    proposal_order = np.zeros(num_players, int)
    arm_matching = [[] for _ in range(num_arms)]

    i = 0
    while np.sum(matched) != num_arms:
        for player in range(num_players):
            if not matched[player]:
                player_proposal = player_ranking[player][proposal_order[player]]
                proposal_order[player] += 1

                if arm_matching[player_proposal] == []:
                    arm_matching[player_proposal] = player
                    matched[player] = True
                elif arms_rankings[player_proposal].index(arm_matching[player_proposal]) > arms_rankings[
                    player_proposal].index(player):
                    matched[arm_matching[player_proposal]] = False
                    matched[player] = True
                    arm_matching[player_proposal] = player
        i += 1
    return inv_matching(arm_matching)