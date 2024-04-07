import argparse
import logging
import os

import numpy as np

from plotting.drawer import draw_gantt_chart
from solution_methods.helper_functions import load_job_shop_env, load_parameters, initialize_device, set_seeds
from solution_methods.L2D.agent_utils import *
from solution_methods.L2D.env_test import NipsJSPEnv_test as Env_test
from solution_methods.L2D.mb_agg import *
from solution_methods.L2D.PPO_model import PPO

PARAM_FILE = "configs/L2D.toml"


def run_method(**parameters):
    # Extract parameters
    device = initialize_device(parameters)
    model_parameters = parameters["network_parameters"]
    train_parameters = parameters["train_parameters"]
    test_parameters = parameters["test_parameters"]
    set_seeds(parameters["test_parameters"]["seed"])

    # Configure default device
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if device.type == 'cuda' else 'torch.FloatTensor')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # Configure environment and load instance
    jobShopEnv = load_job_shop_env(test_parameters["instance_name"])
    env_test = Env_test(n_j=jobShopEnv.nr_of_jobs, n_m=jobShopEnv.nr_of_machines)

    # Load trained policy
    ppo = PPO(train_parameters["lr"], train_parameters["gamma"], train_parameters["k_epochs"], train_parameters["eps_clip"],
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

    trained_policy = os.getcwd() + test_parameters['trained_policy']
    ppo.policy.load_state_dict(torch.load(trained_policy))
    g_pool_step = g_pool_cal(graph_pool_type=model_parameters["graph_pool_type"],
                             batch_size=torch.Size([1, jobShopEnv.nr_of_jobs * jobShopEnv.nr_of_machines, jobShopEnv.nr_of_jobs * jobShopEnv.nr_of_machines]),
                             n_nodes=jobShopEnv.nr_of_jobs * jobShopEnv.nr_of_machines, device=device)

    # run instance
    adj, fea, candidate, mask = env_test.reset(jobShopEnv)
    ep_reward = - env_test.JSM_max_endTime

    while True:
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)

        with torch.no_grad():
            pi, _ = ppo.policy(x=fea_tensor,
                               graph_pool=g_pool_step,
                               padded_nei=None,
                               adj=adj_tensor,
                               candidate=candidate_tensor.unsqueeze(0),
                               mask=mask_tensor.unsqueeze(0))
            if test_parameters['sample']:
                action = sample_select_action(pi, candidate)
            else:
                action = greedy_select_action(pi, candidate)

        adj, fea, reward, done, candidate, mask = env_test.step(action)
        ep_reward += reward

        if done:
            break

    print("makespan:", -ep_reward + env_test.posRewards)
    if test_parameters['plotting']:
        draw_gantt_chart(jobShopEnv)


def main(param_file=PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    run_method(**parameters)


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
