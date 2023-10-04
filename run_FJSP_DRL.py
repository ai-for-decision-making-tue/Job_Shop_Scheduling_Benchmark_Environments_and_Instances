# GITHUB REPO: https://github.com/songwenas12/fjsp-drl

# Code based on the paper:
# "Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning"
# by Wen Song, Xinyang Chen, Qiqiang Li and Zhiguang Cao
# Presented in IEEE Transactions on Industrial Informatics, 2023.
# Paper URL: https://ieeexplore.ieee.org/document/9826438

import argparse
import copy
import logging

import gym
import pynvml
import torch

from solutions.FJSP_DRL import PPO_model
from solutions.FJSP_DRL.env import FJSPEnv
from solutions.FJSP_DRL.load_data import nums_detec
from solutions.helper_functions import load_parameters
from plotting.drawer import draw_gantt_chart

logging.basicConfig(level=logging.INFO)

PARAM_FILE = "configs/FJSP_DRL.toml"
DEFAULT_RESULTS_ROOT = "./results/single_runs"

import pkg_resources
print(pkg_resources.get_distribution("gym").version)



def initialize_device(parameters: dict) -> torch.device:
    device_str = "cpu"
    if parameters['test_parameters']['device'] == "cuda":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def schedule(env: FJSPEnv, model: PPO_model.PPO, memories, flag_sample=False):
    """
    Schedules the provided environment using the given model and memories.

    Parameters:
    - env: The environment to be scheduled.
    - model: The model to be used for scheduling.
    - memories: Memories to be used with the model.
    - flag_sample: A flag to determine whether sampling should be performed.

    Returns:
    - A deep copy of the makespan batch from the environment.
    """

    # Initialize the environment state and completion signals
    state = env.state
    dones = env.done_batch
    done = False

    # Iterate until all tasks are complete
    while not done:
        # Predict the next action without accumulating gradients
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, dones, flag_sample=flag_sample, flag_train=False)

        # Update the environment state based on the predicted action
        state, _, dones = env.step(actions)
        done = dones.all()

    # Draw gantt chart
    for ix, sim_env in enumerate(env.simulation_envs):
        draw_gantt_chart(sim_env)

    # Check the validity of the produced Gantt chart (only feasible for FJSP)
    if not env.validate_gantt()[0]:
        print("Scheduling Error!")

    return copy.deepcopy(env.makespan_batch)


def main(param_file: str):
    # # Initialize NVML library
    pynvml.nvmlInit()

    # Load parameters
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    device = initialize_device(parameters)
    logging.info(f"Using device {device}")
    # Configure PyTorch's default device
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if device.type == 'cuda' else 'torch.FloatTensor')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    # Extract parameters
    env_parameters = parameters["env_parameters"]
    model_parameters = parameters["model_parameters"]
    train_parameters = parameters["train_parameters"]
    test_parameters = parameters["test_parameters"]

    batch_size = test_parameters["num_sample"] if test_parameters["sample"] else 1
    env_parameters["batch_size"] = batch_size

    model_parameters["actor_in_dim"] = model_parameters["out_size_ma"] * 2 + model_parameters["out_size_ope"] * 2
    model_parameters["critic_in_dim"] = model_parameters["out_size_ma"] + model_parameters["out_size_ope"]

    instance_path = "./data/{0}".format(test_parameters["problem_instance"])

    # Assign device to parameters
    env_parameters["device"] = device
    model_parameters["device"] = device

    # Initialize model and environment
    model = PPO_model.PPO(model_parameters, train_parameters)

    # Load trained policy
    trained_policy = test_parameters['trained_policy']
    if trained_policy.endswith('.pt'):
        if device.type == 'cuda':
            policy = torch.load(trained_policy)
        else:
            policy = torch.load(trained_policy, map_location='cpu')
        print('\nloading checkpoint:', trained_policy)
        model.policy.load_state_dict(policy)
        model.policy_old.load_state_dict(policy)

    # Load instance (to configure DRL env)
    with open(instance_path) as file_object:
        line = file_object.readlines()
        num_jobs, num_machines, _ = nums_detec(line)

    env_parameters["num_jobs"] = num_jobs
    env_parameters["num_mas"] = num_machines

    # sampling, each env contains multiple (=num_sample) copies of one instance
    if test_parameters["sample"]:
        env = gym.make('fjsp-v0', case=[instance_path] * test_parameters["num_sample"], env_paras=env_parameters,
                       data_source='file')
    else:
        # greedy, each env contains one instance
        env = gym.make('fjsp-v0', case=[instance_path], env_paras=env_parameters, data_source='file')
    makespan = schedule(env, model, PPO_model.Memory(), flag_sample=test_parameters["sample"])
    print(f"Instance: {instance_path}, Makespan: {makespan}")


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
