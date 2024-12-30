"""
The following code is used to generate the two instances used in the simulations.

Note that in both instances, we use the same seeds for each player-arm pair.
"""
import numpy as np

from matching.cetralised_platforms.pac.BasePlatform import BasePlatform
from matching.matching_algo.gale_shapley import gale_shapley_algo
from setup.setup import RESULTS_PATH
from setup.utils import create_directory, save_json


def constrained_dirichlet_sample(n, k, min_value):
    """
    Generates a sample from a Dirichlet distribution with the constraint that
    each element in the sample is at least `min_value`.

    Parameters:
    - alpha: List of concentration parameters for the Dirichlet distribution.
    - min_value: The minimum value for each element in the sample.
    - size: Number of elements in the sample (i.e., length of the sample).

    Returns:
    - A sample from the constrained Dirichlet distribution.
    """
    # Number of variables

    # Ensure that the sum of the minimum values does not exceed 1
    if (k - 1) * min_value > 1:
        raise ValueError("The minimum value is too large for the number of variables.")

    # Sample from the Dirichlet distribution over the remaining sum (1 - k * min_value)
    remaining_sum = 1 - (k - 1) * min_value
    alpha = np.ones(k - 1)
    dirichlet_sample = np.random.dirichlet(alpha, size=k)

    # Scale the Dirichlet sample to fit within the remaining sum
    scaled_sample = dirichlet_sample * remaining_sum

    # Add the minimum value to each element of the scaled sample
    final_sample = scaled_sample + min_value

    return final_sample - 0.000000001


def fix_rewards(preferences, rewards, p):
    """
    order generated rewards according to the preferences
    """
    rewards_fixed = np.zeros((p, p))

    for p_i in range(p):
        for i, order in enumerate(preferences[p_i]):
            rewards_fixed[p_i][order] = rewards[p_i][p - i - 1]
    return rewards_fixed


# Example usage
def get_both_rewards_preferences(n, min_val):
    # 1. generate random seed
    seed = np.random.randint(10000)
    # 2. generate preferences --  preferences are the same for both instances,
    # we have different rewards because of the different deltas
    preferences = [range(n) for _ in range(n)]
    preferences = np.array(preferences)
    list(map(np.random.shuffle, preferences))

    # 3. generate deltas
    deltas = constrained_dirichlet_sample(n=1, k=n, min_value=min_val)
    zeros = np.zeros((n, 1))
    deltas = np.concatenate((zeros, deltas), axis=1)

    # A. Create random rewards setting
    rewards = np.cumsum(deltas, axis=1)
    rewards_fixed = fix_rewards(preferences, rewards, n)
    test_preferences = BasePlatform.preferences_from_rewards(rewards_fixed)
    assert (test_preferences == preferences).all()  # sanity check for rewards to be in correct order
    # B. create decreasing delta setting
    sorted_delta = np.sort(deltas, axis=1)
    sorted_rewards = np.cumsum(sorted_delta, axis=1)
    sorted_rewards_fixed = fix_rewards(preferences, sorted_rewards, n)
    test_sorted_preferences = BasePlatform.preferences_from_rewards(sorted_rewards_fixed)
    assert (test_sorted_preferences == preferences).all()
    assert test_sorted_preferences == test_preferences

    return (rewards_fixed.tolist(), preferences.tolist()), (sorted_rewards_fixed.tolist(), preferences.tolist()), seed


def get_random_rewards_preferences(n, min_val):
    deltas = constrained_dirichlet_sample(n=1, k=n, min_value=min_val)
    zeros = np.zeros((n, 1))
    deltas = np.concatenate((zeros, deltas), axis=1)
    rewards = np.cumsum(deltas, axis=1)
    list(map(np.random.shuffle, rewards))  # shuffle
    preferences = BasePlatform.preferences_from_rewards(rewards)
    return rewards.tolist(), preferences


def generate_instance(num_instances, agents, min_val):
    player_rewards_list_4 = []
    player_preferences_list_4 = []
    player_rewards_list_5 = []
    player_preferences_list_5 = []
    arm_rewards_list = []
    arm_preferences_list = []

    avg_avg_ranking = []
    avg_min_ranking = []
    avg_max_ranking = []
    seeds = []
    for i in range(num_instances):
        # generate instance
        instance_4, instance_5, seed = get_both_rewards_preferences(n=agents, min_val=min_val)

        # store seed
        seeds += [seed]

        # store instance for random deltas
        player_rewards_4, player_preferences_4 = instance_4
        player_rewards_list_4 += [player_rewards_4]
        player_preferences_list_4 += [player_preferences_4]

        # store instance for decreasing deltas
        player_rewards_5, player_preferences_5 = instance_5
        player_rewards_list_5 += [player_rewards_5]
        player_preferences_list_5 += [player_preferences_5]

        arm_rewards, arm_preferences = get_random_rewards_preferences(n=agents, min_val=min_val)
        arm_rewards_list += [arm_rewards]
        arm_preferences_list += [arm_preferences]

        # here we just print some statics about the instances
        gs_match_4 = gale_shapley_algo(player_ranking=player_preferences_4,
                                       arms_rankings=arm_preferences,
                                       num_players=agents,
                                       num_arms=agents)

        ranks4 = []
        for p in range(agents):
            player_stable_match = gs_match_4[p]
            rank_sm_p4 = player_preferences_4[p].index(player_stable_match)
            ranks4 += [rank_sm_p4]
        avg_avg_ranking += [np.mean(ranks4)]
        avg_min_ranking += [np.min(ranks4)]
        avg_max_ranking += [np.max(ranks4)]
    print(f"player {agents} avg : {np.mean(avg_avg_ranking)}")
    print(f"player {agents} max : {np.max(avg_max_ranking)}")
    print(f"player {agents} min : {np.min(avg_min_ranking)}")

    return (player_rewards_list_4, player_preferences_list_4, arm_rewards_list, arm_preferences_list), \
        (player_rewards_list_5, player_preferences_list_5, arm_rewards_list, arm_preferences_list), \
        seeds


if __name__ == "__main__":
    workspace_dir = RESULTS_PATH + ("/paper/simulations")
    create_directory(workspace_dir)
    for player in [3, 5, 10, 15, 20]:
        instance_1, instance_2, seeds = generate_instance(
            num_instances=100,
            agents=player,
            min_val=0.05)
        # unfold instances
        (player_rewards_4, player_preferences_4, arm_rewards_4, arm_preferences_4) = instance_1
        (player_rewards_5, player_preferences_5, arm_rewards_5, arm_preferences_5) = instance_2

        # save instance 1
        save_data_path = workspace_dir + f"/data/instance_1_player_{player}"
        create_directory(save_data_path)
        save_json(player_rewards_4, save_data_path + "/mean_player_rewards.json")
        save_json(player_preferences_4, save_data_path + "/players_rankings.json")
        save_json(arm_rewards_4, save_data_path + "/mean_arm_rewards.json")
        save_json(arm_preferences_4, save_data_path + "/arm_rankings.json")
        save_json(seeds, save_data_path + "/seeds.json")

        # save instance 2
        save_data_path = workspace_dir + f"/data/instance_2_player_{player}"
        create_directory(save_data_path)
        save_json(player_rewards_5, save_data_path + "/mean_player_rewards.json")
        save_json(player_preferences_5, save_data_path + "/players_rankings.json")
        save_json(arm_rewards_5, save_data_path + "/mean_arm_rewards.json")
        save_json(arm_preferences_5, save_data_path + "/arm_rankings.json")
        save_json(seeds, save_data_path + "/seeds.json")
