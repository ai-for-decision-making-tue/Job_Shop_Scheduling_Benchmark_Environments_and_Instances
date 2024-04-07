import argparse
import logging
import os
import time

import torch

from plotting.drawer import draw_gantt_chart
from solution_methods.FJSP_DRL.env_test import FJSPEnv_test
from solution_methods.FJSP_DRL.PPO import HGNNScheduler
from solution_methods.helper_functions import load_job_shop_env, load_parameters, initialize_device, set_seeds

PARAM_FILE = "configs/FJSP_DRL.toml"


def run_method(**parameters):
    # Extract parameters
    device = initialize_device(parameters)
    model_parameters = parameters["model_parameters"]
    test_parameters = parameters["test_parameters"]
    set_seeds(parameters["test_parameters"]["seed"])

    # Configure default device
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if device.type == 'cuda' else 'torch.FloatTensor')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # Load trained policy
    trained_policy = os.getcwd() + test_parameters['trained_policy']
    if trained_policy.endswith('.pt'):
        if device.type == 'cuda':
            policy = torch.load(trained_policy)
        else:
            policy = torch.load(trained_policy, map_location='cpu')

        model_parameters["actor_in_dim"] = model_parameters["out_size_ma"] * 2 + model_parameters["out_size_ope"] * 2
        model_parameters["critic_in_dim"] = model_parameters["out_size_ma"] + model_parameters["out_size_ope"]

        hgnn_model = HGNNScheduler(model_parameters).to(device)
        print('\nloading saved model:', trained_policy)
        hgnn_model.load_state_dict(policy)

    # Configure environment and load instance
    jobShopEnv = load_job_shop_env(test_parameters['problem_instance'])
    env_test = FJSPEnv_test(jobShopEnv, test_parameters)

    # Get state and completion signal
    state = env_test.state
    done = False
    last_time = time.time()

    # Generate schedule for instance
    while ~done:
        with torch.no_grad():
            actions = hgnn_model.act(state, [], done, flag_train=False, flag_sample=test_parameters['sample'])
        state, _, done = env_test.step(actions)

    print("spend_time:", time.time() - last_time)
    print("makespan(s):", env_test.JSP_instance.makespan)

    if test_parameters['plotting']:
        draw_gantt_chart(env_test.JSP_instance)


def main(param_file=PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    run_method(**parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FJSP_DRL")
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
