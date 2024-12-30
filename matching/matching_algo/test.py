from matching.matching_algo.find_all_matchings import find_all_matchings
from matching.utils import inv_matching, fix_preferences, calculate_delta_rank
from matching.matching_algo.find_all_stable_matching_brute_force import all_stable_matching_brute_force
from matching.matching_algo.gale_shapley import gale_shapley_algo
from matching.matching_algo.is_stable import is_unstable
import numpy as np


def test_1():
    players_ranking_test = [[0, 1], [0, 1]]
    arm_ranking_test = [[0, 1], [0, 1]]
    num_players_test = 2

    print("--------------------------------")
    print("Test stability of matching [1,0]")
    arm_matching_test = [1, 0]
    res = is_unstable(arm_matching=arm_matching_test,
                      players_ranking=players_ranking_test,
                      arms_rankings=arm_ranking_test,
                      num_players=num_players_test)
    token = "unstable" if res else "stable"
    print("The matching is " + str(token))
    assert res == True

    print("--------------------------------")
    print("Test stability of all matchings")
    matchings = find_all_matchings(num_of_players=2)
    for matching in matchings:
        matching = list(matching)
        res = is_unstable(arm_matching=matching,
                          players_ranking=players_ranking_test,
                          arms_rankings=arm_ranking_test,
                          num_players=num_players_test)
        print(res)
        token = "unstable" if res else "stable"
        print(f"The matching {matching} is " + str(token))

    print("--------------------------------")
    print("Test find all stable matching")
    matching = all_stable_matching_brute_force(players_ranking=players_ranking_test,
                                               arms_rankings=arm_ranking_test,
                                               num_players=num_players_test)
    print(f"The matching: {matching}")
    print(len(matching))

    print("End")


# test_1()
def test_2():
    """
    example from 3.1 from Procedurally fair and stable matching paper B. Klaus and F. Klijn
    """
    players_ranking_test = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
    arm_ranking_test = [[2, 1, 0], [1, 0, 2], [0, 2, 1]]
    num_players_test = 3
    matchings = all_stable_matching_brute_force(players_ranking=players_ranking_test,
                                                arms_rankings=arm_ranking_test,
                                                num_players=num_players_test)
    for matching in matchings:
        print(np.array(inv_matching(matching)) + 1)
    assert len(matchings) == 3

    print("find Men optimal matching")
    optimal_men_matching = gale_shapley_algo(arms_rankings=arm_ranking_test,
                                             player_ranking=players_ranking_test,
                                             num_players=num_players_test,
                                             num_arms=num_players_test)
    print(np.array(optimal_men_matching) + 1)
    print("find Woman optimal matching")
    woman_men_matching = gale_shapley_algo(arms_rankings=players_ranking_test,
                                           player_ranking=arm_ranking_test,
                                           num_players=num_players_test,
                                           num_arms=num_players_test)
    woman_men_matching = inv_matching(woman_men_matching)
    print(np.array(woman_men_matching) + 1)
    print("End")


def test_3():
    """
    example from 3.1 from Procedurally fair and stable matching paper
    """
    num_players_test = 5

    players_ranking_test = [[1, 3, 2, 4, 5],
                            [2, 3, 1, 4, 5],
                            [3, 2, 1, 4, 5],
                            [4, 5, 1, 2, 3],
                            [5, 4, 1, 2, 3]]
    players_ranking_test = (np.array(players_ranking_test) - 1).tolist()
    print(players_ranking_test)

    arm_ranking_test = [[2, 1, 3, 4, 5],
                        [3, 2, 1, 4, 5],
                        [1, 2, 3, 4, 5],
                        [5, 4, 1, 2, 3],
                        [4, 5, 1, 2, 3]]
    arm_ranking_test = (np.array(arm_ranking_test) - 1).tolist()
    print(arm_ranking_test)

    matchings = all_stable_matching_brute_force(players_ranking=players_ranking_test,
                                                arms_rankings=arm_ranking_test,
                                                num_players=num_players_test)
    print("All matchings")
    for matching in matchings:
        delta = calculate_delta_rank(matching=matching,
                                     true_player_preferences=players_ranking_test,
                                     true_arm_preferences=arm_ranking_test)
        print(np.array(inv_matching(matching)) + 1, "-->", delta)
    assert len(matchings) == 6

    print("find Men optimal matching")
    optimal_men_matching = gale_shapley_algo(arms_rankings=arm_ranking_test,
                                             player_ranking=players_ranking_test,
                                             num_players=num_players_test,
                                             num_arms=num_players_test)
    print(np.array(optimal_men_matching) + 1)
    print("find Woman optimal matching")
    woman_men_matching = gale_shapley_algo(arms_rankings=players_ranking_test,
                                           player_ranking=arm_ranking_test,
                                           num_players=num_players_test,
                                           num_arms=num_players_test)
    woman_men_matching = inv_matching(woman_men_matching)
    print(np.array(woman_men_matching) + 1)
    print("End")


def test_bug():
    # ucb
    # est
    # [[3.79254901e+00 1.01058681e+00 1.00000000e+02]
    #  [1.00000000e+02 2.57375354e+00 1.04314599e-02]
    # [2.02955691e+00
    # 1.00000000e+02
    # 2.96803067e+00]]
    # ucb
    # pref
    arm_ranking = [list(range(1, 3 + 1))] * 3
    arm_ranking = fix_preferences(arm_ranking)
    pref = [[2, 0, 1], [0, 1, 2], [1, 2, 0]]
    bug_match = [1, 2, 0]
    gale_shapley_match = gale_shapley_algo(player_ranking=pref,
                                           arms_rankings=arm_ranking,
                                           num_players=3,
                                           num_arms=3)
    print("pref")
    print(pref)
    print("arm_ranking")
    print(arm_ranking)
    print("bug_match")
    print(bug_match)
    print("gale_shapley_match")
    print(gale_shapley_match)

    print("all matchings")
    matchings = all_stable_matching_brute_force(players_ranking=pref,
                                                arms_rankings=arm_ranking,
                                                num_players=3)
    for matching in matchings:
        print(matching)

    # check match
    my_match = [2, 0, 1]
    unstable = is_unstable(arm_matching=my_match,
                           players_ranking=arm_ranking,
                           arms_rankings=pref,
                           num_players=3)
    print(unstable)


test_3()
