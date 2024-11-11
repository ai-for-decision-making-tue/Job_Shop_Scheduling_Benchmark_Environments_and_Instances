# GITHUB REPO: https://github.com/zcaicaros/L2D

# Code based on the paper:
# Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning"
# by Cong Zhang, Wen Song, Zhiguang Cao, Jie Zhang, Puay Tan and Xu Chi
# Presented in 34th Conference on Neural Information Processing Systems (NeurIPS), 2020.
# Paper URL: https://papers.nips.cc/paper_files/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf

import argparse
import logging
import os
import numpy as np
import torch

from plotting.drawer import plot_gantt_chart
from solution_methods.helper_functions import load_job_shop_env, load_parameters, initialize_device, set_seeds
from solution_methods.L2D.src.agent_utils import sample_select_action, greedy_select_action
from solution_methods.L2D.src.env_test import NipsJSPEnv_test as Env_test
from solution_methods.L2D.src.mb_agg import g_pool_cal
from solution_methods.L2D.src.PPO_model import PPO
from utils import output_dir_exp_name, results_saving

PARAM_FILE = "../../configs/L2D.toml"
logging.basicConfig(level=logging.INFO)


def run_L2D(jobShopEnv, **parameters):
    # Set up device and seeds
    device = initialize_device(parameters)
    set_seeds(parameters["test_parameters"]["seed"])

    # Configure default tensor type for device
    torch.set_default_device('cuda' if device.type == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # Configure test environment
    env_test = Env_test(n_j=jobShopEnv.nr_of_jobs, n_m=jobShopEnv.nr_of_machines)

    # Initialize PPO model with network and training parameters
    model_parameters = parameters["network_parameters"]
    train_parameters = parameters["train_parameters"]
    ppo = PPO(lr=train_parameters["lr"],
              gamma=train_parameters["gamma"],
              k_epochs=train_parameters["k_epochs"],
              eps_clip=train_parameters["eps_clip"],
              n_j=jobShopEnv.nr_of_jobs,
              n_m=jobShopEnv.nr_of_machines,
              num_layers=model_parameters["num_layers"],
              neighbor_pooling_type=model_parameters["neighbor_pooling_type"],
              input_dim=model_parameters["input_dim"],
              hidden_dim=model_parameters["hidden_dim"],
              num_mlp_layers_feature_extract=model_parameters["num_mlp_layers_feature_extract"],
              num_mlp_layers_actor=model_parameters["num_mlp_layers_actor"],
              hidden_dim_actor=model_parameters["hidden_dim_actor"],
              num_mlp_layers_critic=model_parameters["num_mlp_layers_critic"],
              hidden_dim_critic=model_parameters["hidden_dim_critic"])

    # Load trained policy
    trained_policy = os.path.dirname(os.path.abspath(__file__)) + parameters['test_parameters'].get('trained_policy')
    ppo.policy.load_state_dict(torch.load(trained_policy, map_location=torch.device(parameters['test_parameters']['device']), weights_only=True))
    logging.info(f"Trained policy loaded from {parameters['test_parameters'].get('trained_policy')}.")

    # Initialize graph pooling step
    g_pool_step = g_pool_cal(graph_pool_type=model_parameters["graph_pool_type"],
                             batch_size=torch.Size([1, jobShopEnv.nr_of_jobs * jobShopEnv.nr_of_machines, jobShopEnv.nr_of_jobs * jobShopEnv.nr_of_machines]),
                             n_nodes=jobShopEnv.nr_of_jobs * jobShopEnv.nr_of_machines,
                             device=device)

    # Run environment instance
    adj, fea, candidate, mask = env_test.reset(jobShopEnv)
    ep_reward = - env_test.JSM_max_endTime

    while True:
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)

        with torch.no_grad():
            pi, _ = ppo.policy(
                x=fea_tensor,
                graph_pool=g_pool_step,
                padded_nei=None,
                adj=adj_tensor,
                candidate=candidate_tensor.unsqueeze(0),
                mask=mask_tensor.unsqueeze(0))

            # Choose action based on sampling or greedy selection
            action = (sample_select_action(pi, candidate) if parameters["test_parameters"]["sample"]
                      else greedy_select_action(pi, candidate))

        # Perform action and observe results
        adj, fea, reward, done, candidate, mask = env_test.step(action)
        ep_reward += reward

        if done:
            break

    makespan = float(-ep_reward + env_test.posRewards)
    logging.info(f"Makespan: {makespan}")

    return makespan, jobShopEnv


def main(param_file=PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    jobShopEnv = load_job_shop_env(parameters['test_parameters'].get('problem_instance'))
    makespan, jobShopEnv = run_L2D(jobShopEnv, **parameters)

    if makespan is not None:
        # Check output configuration and prepare output paths if needed
        output_config = parameters['test_parameters']
        save_gantt = output_config.get('save_gantt')
        save_results = output_config.get('save_results')
        show_gantt = output_config.get('show_gantt')

        if save_gantt or save_results:
            output_dir, exp_name = output_dir_exp_name(parameters)
            output_dir = os.path.join(output_dir, f"{exp_name}")
            os.makedirs(output_dir, exist_ok=True)

        # Plot Gantt chart if required
        if show_gantt or save_gantt:
            logging.info("Generating Gantt chart.")
            plt = plot_gantt_chart(jobShopEnv)

            if save_gantt:
                plt.savefig(output_dir + "\gantt.png")
                logging.info(f"Gantt chart saved to {output_dir}")

            if show_gantt:
                plt.show()

        # Save results if enabled
        if save_results:
            results_saving(makespan, output_dir, parameters)
            logging.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run L2D")
    parser.add_argument(
        "config_file",
        metavar='-f',
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)
