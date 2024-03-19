import random
import time
import sys
import torch
import argparse
import logging
import numpy as np
from pathlib import Path

from solution_methods.helper_functions import load_parameters, load_job_shop_env
from solution_methods.FJSP_DRL.env_test import FJSPEnv_test
from solution_methods.FJSP_DRL import PPO_model as PPO_model
from plotting.drawer import draw_gantt_chart

# Add the base path to the Python module search path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

# import parameter settings
PARAM_FILE = str(base_path) + "/configs/FJSP_DRL.toml"


def initialize_device(parameters: dict) -> torch.device:
    device_str = "cpu"
    if parameters['test_parameters']['device'] == "cuda":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def test_instance(param_file: str = PARAM_FILE):
    # print(os.path.abspath('..'))
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

    # Extract parameters
    env_parameters = parameters["env_parameters"]
    model_parameters = parameters["model_parameters"]
    train_parameters = parameters["train_parameters"]
    test_parameters = parameters["test_parameters"]

    model_parameters["actor_in_dim"] = model_parameters["out_size_ma"] * 2 + model_parameters["out_size_ope"] * 2
    model_parameters["critic_in_dim"] = model_parameters["out_size_ma"] + model_parameters["out_size_ope"]

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_parameters, train_parameters)

    instance_path = test_parameters['problem_instance']
    JobShop_Instance = load_job_shop_env(instance_path)
    env_parameters['num_jobs'] = JobShop_Instance.jobs.__len__()
    env_parameters['num_mas'] = JobShop_Instance.machines.__len__()
    env_parameters['batch_size'] = test_parameters['num_sample']

    # Load trained policy
    trained_policy = str(base_path) + test_parameters['trained_policy']
    if trained_policy.endswith('.pt'):
        if device.type == 'cuda':
            policy = torch.load(trained_policy)
        else:
            policy = torch.load(trained_policy, map_location='cpu')
        print('\nloading saved model:', trained_policy)
        model.policy.load_state_dict(policy)
        model.policy_old.load_state_dict(policy)

    JSMEnvs = [load_job_shop_env(instance_path) for _ in range(test_parameters['num_sample'])]
    for each_env in JSMEnvs:
        each_env.reset()
    env_test = FJSPEnv_test(JSMEnvs, env_parameters)

    # Get state and completion signal
    state = env_test.state
    done = False
    dones = env_test.done_batch
    last_time = time.time()

    # Schedule in parallel
    while ~done:
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, dones, flag_train=False, flag_sample=True)
        state, _, dones = env_test.step(actions)
        done = dones.all()
        # gpu_tracker.track()  # Used to monitor memory (of gpu)
    print("spend_time:", time.time() - last_time)
    print("makespan(s):", [env_test.JSP_instance[i].makespan for i in range(env_test.batch_size)])

    if test_parameters['plotting']:
        for i in range(env_test.batch_size):
            draw_gantt_chart(env_test.JSP_instance[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test FJSP_DRL")
    parser.add_argument(
        "config_file",
        metavar='-f',
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )
    setup_seed(1)
    args = parser.parse_args()
    test_instance(param_file=args.config_file)