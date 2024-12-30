"""
Utility functions for the matching problem.
"""

import random

import numpy as np


def fix_preferences(preferences):
    preferences_fixed = (np.array(preferences) - 1).tolist()
    return preferences_fixed


def inv_matching(matching):
    num_players = len(matching)
    inv_match = np.ones(num_players, int) * (-1)
    for player, arm in enumerate(matching):
        inv_match[arm] = player
    return inv_match


def matching_to_tuples(matching):
    tuple_matching = [(p, a) for p, a in enumerate(matching)]
    return tuple_matching


def calculate_delta_rank(matching, true_player_preferences, true_arm_preferences):
    rank_player = 0
    rank_arm = 0
    for player, arm in enumerate(matching):
        rank_player += true_player_preferences[player].index(arm)
        rank_arm += true_arm_preferences[arm].index(player)
    delta = rank_player - rank_arm
    return delta


def get_rank_function(preferences):
    pref = np.array(preferences)
    rank = np.zeros(pref.shape)
    for player in range(pref.shape[0]):
        for position, arm in enumerate(pref[player, :]):
            rank[player, arm] = position
    return rank

