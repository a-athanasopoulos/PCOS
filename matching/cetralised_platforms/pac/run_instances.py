"""
Here we iteratively run an algorithm for several runs each with a unique instance according to the setting
"""
import time
from matching.cetralised_platforms.pac.cetralised_platform import CentralisedPlatform
import pandas as pd
from setup.utils import create_directory, save_json


def run_instances(algorithm,
                  delta,
                  true_players_rankings,
                  true_arm_rankings,
                  mean_rewards_players,
                  mean_rewards_arms,
                  num_players,
                  variance,
                  seeds,
                  save_path):
    results = []
    instances = len(true_players_rankings)
    for id in range(instances):
        print("try id", id)
        start_time = time.time()
        algo: CentralisedPlatform = algorithm(delta=delta,
                                              player_preferences=true_players_rankings[id],
                                              arm_preferences=true_arm_rankings[id],
                                              mean_rewards_players=mean_rewards_players[id],
                                              mean_rewards_arm=mean_rewards_arms[id],
                                              num_players=num_players,
                                              variance=variance,
                                              seed=seeds[id],
                                              random_state=None)

        algo.run()
        create_directory(save_path + f"/id_{id}")
        print(f"id {id} time: {time.time() - start_time}")
        print("")
        algo.save_results(save_path + f"/id_{id}")

        # get metrics
        results += [{
            "optimal_stable": algo.optimal_stable,
            "stable:": algo.stability,
            "sample_complexity": algo.sample_complexity,
            "rounds": algo.round,
            "samples": algo.num_samples,
            "max_pref": algo.max_pref
        }]

        if id == 1:
            save_json(algo.collected_samples, save_path + '/collected_samples.json')
            # pd.DataFrame(algo.collected_samples).to_csv(save_path + '/collected_samples.csv')
    # save
    results = pd.DataFrame(results)
    results.to_csv(save_path + '/results.csv')
    return results
