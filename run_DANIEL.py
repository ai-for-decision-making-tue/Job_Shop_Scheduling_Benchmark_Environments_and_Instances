import argparse
import logging
import os
import time

import torch

from plotting.drawer import draw_gantt_chart
from solution_methods.DANIEL.common_utils import greedy_select_action, sample_action
from solution_methods.DANIEL.env_test import FJSPEnv_test
from solution_methods.DANIEL.model.PPO import PPO_initialize
from solution_methods.helper_functions import (
    initialize_device,
    load_job_shop_env,
    load_parameters,
    set_seeds,
)

PARAM_FILE = "configs/DANIEL.toml"


def run_method(**parameters):
    # Extract parameters
    device = initialize_device(parameters, method="DANIEL")
    set_seeds(parameters["testing"]["seed"])

    # Configure default device
    torch.set_default_tensor_type(
        "torch.cuda.FloatTensor" if device.type == "cuda" else "torch.FloatTensor"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    agent = PPO_initialize(parameters)
    # Load trained policy
    trained_policy = (
        os.getcwd() + f"/solution_methods/DANIEL/save/{parameters['model']['source']}"
        f"/{parameters['testing']['test_model']}.pth"
    )
    if device.type == "cuda":
        policy = torch.load(trained_policy, map_location="cuda")
    else:
        policy = torch.load(trained_policy, map_location="cpu")

    print("\nloading saved model:", trained_policy)
    agent.policy.load_state_dict(policy)
    agent.policy.eval()

    # Configure environment and load instance
    jobShopEnv = load_job_shop_env(parameters["testing"]["problem_instance"])
    env_test = FJSPEnv_test(jobShopEnv, parameters)

    # Get state and completion signal
    state = env_test.state
    done = False
    last_time = time.time()
    count = 0
    # Generate schedule for instance
    while not done:
        with torch.no_grad():
            pi, _ = agent.policy(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
            )
        if not parameters["testing"]["sample"]:
            action = greedy_select_action(pi)
        else:
            action, _ = sample_action(pi)

        state, reward, done = env_test.step(actions=action.cpu().numpy())
        count += 1
    print("spend_time:", time.time() - last_time)
    print("makespan(s):", env_test.JSP_instance.makespan)

    if parameters["testing"]["plotting"]:
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
        metavar="-f",
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)
