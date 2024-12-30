"""
This code runs the simulations both instance and every algorithm and number of agents
"""
from matching.cetralised_platforms.pac.get_algo import pac_algorithms
from matching.cetralised_platforms.pac.run_instances import run_instances
from setup.setup import RESULTS_PATH
from setup.utils import create_directory, load_json


def run_algorithms(delta,
                   algos,
                   num_players,
                   save_path,
                   players_rankings,
                   arm_rankings,
                   mean_player_rewards,
                   mean_arm_rewards,
                   seeds,
                   variance=1):
    """"
    runs all algorithm for on a list of instances
    """
    for algorithm_name in algos:
        print(f"RUN {algorithm_name}")
        save_dir_algos = save_path + f"/{algorithm_name}"
        save_dir_delta = save_dir_algos + f"/players_{num_players}_delta_{delta}"
        create_directory(save_dir_delta)

        algorithm = pac_algorithms[algorithm_name]
        results = run_instances(algorithm=algorithm,
                                delta=delta,
                                true_players_rankings=players_rankings,
                                true_arm_rankings=arm_rankings,
                                mean_rewards_players=mean_player_rewards,
                                mean_rewards_arms=mean_arm_rewards,
                                num_players=num_players,
                                variance=variance,
                                seeds=seeds,
                                save_path=save_dir_delta)


if __name__ == "__main__":
    # workspace has to be the same with one we used when we generate the instance
    workspace_dir = RESULTS_PATH + "/paper/simulations"
    create_directory(workspace_dir)

    algos = ["NaiveUniformlySampling", "Elimination", "ImprovedElimination", "AdapElimination"]
    delta = 0.1  # δ
    for num_p in [3, 5, 10, 15, 20]:
        for instance in [1, 2]:
            load_data_path = workspace_dir + f"/data/instance_{instance}_player_{num_p}"

            players_rankings = load_json(load_data_path + "/players_rankings.json")
            arm_rankings = load_json(load_data_path + "/arm_rankings.json")
            mean_player_rewards = load_json(load_data_path + "/mean_player_rewards.json")
            mean_arm_rewards = load_json(load_data_path + "/mean_arm_rewards.json")
            seeds = load_json(load_data_path + "/seeds.json")
            print("**************************")
            print(f"** Run instance {instance} for delta {delta} num_players {num_p} **")
            print("**************************")
            run_algorithms(delta=delta,
                           algos=algos,
                           num_players=num_p,
                           save_path=workspace_dir + f"/instance_{instance}/run_1/",
                           players_rankings=players_rankings,
                           arm_rankings=arm_rankings,
                           mean_player_rewards=mean_player_rewards,
                           mean_arm_rewards=mean_arm_rewards,
                           seeds=seeds,
                           variance=1)
