import argparse
import copy
import logging
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from visdom import Visdom

import solution_methods.FJSP_DRL.PPO as PPO_model
from solution_methods.FJSP_DRL.case_generator import CaseGenerator
from solution_methods.FJSP_DRL.env_training import FJSPEnv_training
from solution_methods.FJSP_DRL.validate import get_validate_env, validate
from solution_methods.helper_functions import load_parameters

# Add the base path to the Python module search path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

# import FJSP parameters
PARAM_FILE = str(base_path) + "/configs/FJSP_DRL.toml"


def initialize_device(parameters: dict) -> torch.device:
    device_str = "cpu"
    if parameters["test_parameters"]["device"] == "cuda":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def train_FJSP_DRL(param_file: str = PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    device = initialize_device(parameters)

    # Configure PyTorch's default device
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if device.type == 'cuda' else 'torch.FloatTensor')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # Extract parameters
    env_parameters = parameters["env_parameters"]
    model_parameters = parameters["model_parameters"]
    train_parameters = parameters["train_parameters"]

    env_validation_parameters = copy.deepcopy(env_parameters)
    env_validation_parameters["batch_size"] = env_parameters["valid_batch_size"]

    model_parameters["actor_in_dim"] = model_parameters["out_size_ma"] * 2 + model_parameters["out_size_ope"] * 2
    model_parameters["critic_in_dim"] = model_parameters["out_size_ma"] + model_parameters["out_size_ope"]

    num_jobs = env_parameters["num_jobs"]
    num_machines = env_parameters["num_mas"]
    opes_per_job_min = int(num_machines * 0.8)
    opes_per_job_max = int(num_machines * 1.2)
    print(num_jobs, num_machines)

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_parameters, train_parameters, num_envs=env_parameters["batch_size"])

    env_valid = get_validate_env(env_validation_parameters, train_parameters)  # Create an environment for validation
    maxlen = 1  # Save the best model
    best_models = deque()
    makespan_best = float("inf")

    # Use visdom to visualize the training process
    is_viz = train_parameters["viz"]
    if is_viz:
        viz = Visdom(env=train_parameters["viz_name"])

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = "./save/train_{0}".format(str_time)
    os.makedirs(save_path)

    valid_results = []
    valid_results_100 = []

    # Training part
    env_training = None
    for i in range(1, train_parameters["max_iterations"] + 1):
        if (i - 1) % train_parameters["parallel_iter"] == 0:
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_machines, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
            env_training = FJSPEnv_training(case=case, env_paras=env_parameters)

        # Get state and completion signal
        env_training.reset()
        state = env_training.state
        done = False
        dones = env_training.done_batch
        last_time = time.time()

        # Schedule in parallel
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            state, rewards, dones, _ = env_training.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
            # gpu_tracker.track()  # Used to monitor memory (of gpu)
        print("spend_time: ", time.time() - last_time)

        # Verify the solution
        gantt_result = env_training.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")
        # print("Scheduling Finish")
        env_training.reset()

        # if iter mod x = 0 then update the policy (x = 1 in paper)
        if i % train_parameters["update_timestep"] == 0:
            loss, reward = model.update(memories, env_parameters, train_parameters)
            print("reward: ", "%.3f" % reward, "; loss: ", "%.3f" % loss)
            memories.clear_memory()

            if is_viz:
                viz.line(X=np.array([i]), Y=np.array([reward]),
                         win='window{}'.format(0), update='append', opts=dict(title='reward of envs'))
                viz.line(X=np.array([i]), Y=np.array([loss]),
                         win='window{}'.format(1), update='append', opts=dict(title='loss of envs'))

        # if iter mod x = 0 then validate the policy (x = 10 in paper)
        if i % train_parameters["save_timestep"] == 0:
            print("\nStart validating")
            # Record the average results and the results on each instance
            vali_result, vali_result_100 = validate(env_validation_parameters, env_valid, model.policy_old)
            valid_results.append(vali_result.item())
            valid_results_100.append(vali_result_100)

            # Save the best model
            if vali_result < makespan_best:
                makespan_best = vali_result
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = "{0}/save_best_{1}_{2}_{3}.pt".format(save_path, num_jobs, num_machines, i)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

            if is_viz:
                viz.line(
                    X=np.array([i]), Y=np.array([vali_result.item()]),
                    win="window{}".format(2), update="append", opts=dict(title="makespan of valid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train FJSP_DRL")
    parser.add_argument(
        "config_file",
        metavar="-f",
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    train_FJSP_DRL(param_file=args.config_file)
