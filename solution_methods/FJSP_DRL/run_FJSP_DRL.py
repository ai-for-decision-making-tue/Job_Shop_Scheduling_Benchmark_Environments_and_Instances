# GITHUB REPO: https://github.com/songwenas12/fjsp-drl

# Code based on the paper:
# "Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning"
# by Wen Song, Xinyang Chen, Qiqiang Li and Zhiguang Cao
# Presented in IEEE Transactions on Industrial Informatics, 2023.
# Paper URL: https://ieeexplore.ieee.org/document/9826438

import argparse
import logging
import os
import torch

from plotting.drawer import plot_gantt_chart
from solution_methods.helper_functions import load_job_shop_env, load_parameters, initialize_device, set_seeds
from solution_methods.FJSP_DRL.src.env_test import FJSPEnv_test
from solution_methods.FJSP_DRL.src.PPO import HGNNScheduler
from solution_methods.FJSP_DRL.utils import output_dir_exp_name, results_saving

PARAM_FILE = "../../configs/FJSP_DRL.toml"
logging.basicConfig(level=logging.INFO)


def run_FJSP_DRL(jobShopEnv, **parameters):
    # Set up device and seeds
    device = initialize_device(parameters)
    set_seeds(parameters["test_parameters"]["seed"])

    # Configure default tensor type for device
    torch.set_default_device('cuda' if device.type == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # Configure environment and load instance
    env_test = FJSPEnv_test(jobShopEnv, parameters["test_parameters"])

    # Load trained policy
    model_parameters = parameters["model_parameters"]
    test_parameters = parameters["test_parameters"]
    trained_policy = os.path.dirname(os.path.abspath(__file__)) + test_parameters['trained_policy']
    if trained_policy.endswith('.pt'):
        if device.type == 'cuda':
            policy = torch.load(trained_policy)
        else:
            policy = torch.load(trained_policy, map_location='cpu', weights_only=True)

        logging.info(f"Trained policy loaded from {test_parameters.get('trained_policy')}.")
        model_parameters["actor_in_dim"] = model_parameters["out_size_ma"] * 2 + model_parameters["out_size_ope"] * 2
        model_parameters["critic_in_dim"] = model_parameters["out_size_ma"] + model_parameters["out_size_ope"]

        hgnn_model = HGNNScheduler(model_parameters).to(device)
        print('\nloading saved model:', trained_policy)
        hgnn_model.load_state_dict(policy)

    # Get state and completion signal
    state = env_test.state
    done = False

    # Generate schedule for instance
    while ~done:
        with torch.no_grad():
            actions = hgnn_model.act(state, [], done, flag_train=False, flag_sample=test_parameters['sample'])
        state, _, done = env_test.step(actions)

    makespan = env_test.JSP_instance.makespan
    logging.info(f"Makespan: {makespan}")

    return makespan, jobShopEnv


def main(param_file=PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    jobShopEnv = load_job_shop_env(parameters['test_parameters']['problem_instance'])
    makespan, jobShopEnv = run_FJSP_DRL(jobShopEnv, **parameters)

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
                plt.savefig(output_dir + "/gantt.png")
                logging.info(f"Gantt chart saved to {output_dir}")

            if show_gantt:
                plt.show()

        # Save results if enabled
        if save_results:
            results_saving(makespan, output_dir, parameters)
            logging.info(f"Results saved to {output_dir}")


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
