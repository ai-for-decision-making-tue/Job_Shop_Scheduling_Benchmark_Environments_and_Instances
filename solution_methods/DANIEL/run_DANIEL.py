import argparse
import logging
import os
import torch

from plotting.drawer import plot_gantt_chart
from solution_methods.helper_functions import load_job_shop_env, load_parameters, initialize_device, set_seeds
from solution_methods.DANIEL.src.common_utils import greedy_select_action, sample_action
from solution_methods.DANIEL.src.env_test import FJSPEnv_test
from solution_methods.DANIEL.network.PPO import PPO_initialize
from solution_methods.DANIEL.utils import output_dir_exp_name, results_saving

PARAM_FILE = "../../configs/DANIEL.toml"
logging.basicConfig(level=logging.INFO)


def run_DANIEL_FJSP(jobShopEnv, **parameters):
    # Set up device and seeds
    device = initialize_device(parameters, method="DANIEL")
    set_seeds(parameters["test_parameters"]["seed"])

    # Configure default device
    torch.set_default_device('cuda' if device.type == 'cuda' else 'cpu')
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Configure test environment
    env_test = FJSPEnv_test(jobShopEnv, parameters)

    # Initialize PPO model and load trained model
    ppo = PPO_initialize(parameters)

    # load trained policy
    trained_policy = (os.path.dirname(os.path.abspath(__file__)) + f"/save/{parameters['model']['source']}" f"/{parameters['test_parameters']['trained_policy']}.pth")
    policy = torch.load(trained_policy, map_location=device.type, weights_only=True)
    ppo.policy.load_state_dict(policy)
    ppo.policy.eval()
    logging.info(f"Trained policy loaded from {parameters['test_parameters']['trained_policy']}.")


    # Get state and completion signal
    state = env_test.state

    while True:
        with torch.no_grad():
            pi, _ = ppo.policy(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
            )

        # Choose action based on sampling or greedy selection
        action = (sample_action(pi) if parameters["test_parameters"]["sample"]
                  else greedy_select_action(pi))

        # Perform action
        state, reward, done = env_test.step(actions=action.cpu().numpy())

        if done:
            break

    makespan = env_test.JSP_instance.makespan

    return makespan, env_test.JSP_instance
    logging.info(f"Makespan: {makespan}")


def main(param_file=PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    jobShopEnv = load_job_shop_env(parameters["test_parameters"]["problem_instance"])
    makespan, jobShopEnv = run_DANIEL_FJSP(jobShopEnv, **parameters)

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
        metavar="-f",
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)
