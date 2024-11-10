# GITHUB REPO: https://github.com/zcaicaros/L2D

# Code based on the paper:
# Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning"
# by Cong Zhang, Wen Song, Zhiguang Cao, Jie Zhang, Puay Tan and Xu Chi
# Presented in 34th Conference on Neural Information Processing Systems (NeurIPS), 2020.
# Paper URL: https://papers.nips.cc/paper_files/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import torch

from solution_methods.helper_functions import load_parameters, initialize_device, set_seeds
from solution_methods.L2D.src.agent_utils import select_action
from solution_methods.L2D.src.JSSP_Env import SJSSP
from solution_methods.L2D.src.mb_agg import g_pool_cal
from solution_methods.L2D.src.PPO_model import PPO, Memory
from solution_methods.L2D.data.instance_generator import uniform_instance_generator
from solution_methods.L2D.src.validation import validate

base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

PARAM_FILE = str(base_path) + "/configs/L2D.toml"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def train_L2D(**parameters):
    logging.info("Training started.")  # Log the start of training

    # retrieve parameters for environment, model, and training
    env_parameters = parameters["env_parameters"]
    model_parameters = parameters["network_parameters"]
    train_parameters = parameters["train_parameters"]

    # Set up device and seeds
    device = initialize_device(parameters)
    set_seeds(parameters["test_parameters"]["seed"])

    # Create directories for logging and saving models
    os.makedirs('./training_log', exist_ok=True)
    os.makedirs('./saved_models', exist_ok=True)

    # Configure default tensor type for device
    torch.set_default_device('cuda' if device.type == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # Initialize multiple job shop scheduling environments for training
    n_job = env_parameters["n_j"]
    n_machine = env_parameters["n_m"]
    envs = [SJSSP(n_j=n_job, n_m=n_machine) for _ in range(train_parameters["num_envs"])]

    # Load validation data
    data_generator = uniform_instance_generator
    file_path = str(base_path) + '/solution_methods/L2D/data/'
    dataLoaded = np.load(f"{file_path}generatedData{n_job}_{n_machine}_Seed{env_parameters['np_seed_validation']}.npy")
    vali_data = [(dataLoaded[i][0], dataLoaded[i][1]) for i in range(dataLoaded.shape[0])]

    # Initialize memories for each environment instance
    memories = [Memory() for _ in range(train_parameters["num_envs"])]

    # Instantiate the PPO model
    ppo = PPO(lr=train_parameters["lr"],
              gamma=train_parameters["gamma"],
              k_epochs=train_parameters["k_epochs"],
              eps_clip=train_parameters["eps_clip"],
              n_j=n_job,
              n_m=n_machine,
              num_layers=model_parameters["num_layers"],
              neighbor_pooling_type=model_parameters["neighbor_pooling_type"],
              input_dim=model_parameters["input_dim"],
              hidden_dim=model_parameters["hidden_dim"],
              num_mlp_layers_feature_extract=model_parameters["num_mlp_layers_feature_extract"],
              num_mlp_layers_actor=model_parameters["num_mlp_layers_actor"],
              hidden_dim_actor=model_parameters["hidden_dim_actor"],
              num_mlp_layers_critic=model_parameters["num_mlp_layers_critic"],
              hidden_dim_critic=model_parameters["hidden_dim_critic"])

    # Initialize graph pooling setup
    g_pool_step = g_pool_cal(graph_pool_type=model_parameters["graph_pool_type"],
                             batch_size=torch.Size([1, n_job * n_machine, n_job * n_machine]),
                             n_nodes=n_job * n_machine,
                             device=device)
    # training loop
    log = []
    validation_log = []
    record = 100000

    for i_update in range(train_parameters["max_updates"]):

        # Initialize reward tracking for each environment
        ep_rewards = [0 for _ in range(train_parameters["num_envs"])]
        adj_envs, fea_envs, candidate_envs, mask_envs = [], [], [], []

        for i, env in enumerate(envs):
            adj, fea, candidate, mask = env.reset(data_generator(n_j=n_job,
                                                                 n_m=n_machine,
                                                                 low=env_parameters['low'],
                                                                 high=env_parameters["high"]))
            adj_envs.append(adj)
            fea_envs.append(fea)
            candidate_envs.append(candidate)
            mask_envs.append(mask)
            ep_rewards[i] = - env.initQuality

        # Rollout: interact with envs until all environments are done
        while True:
            # Convert environment states to tensors
            fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(device) for fea in fea_envs]
            adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(device).to_sparse() for adj in adj_envs]
            candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(device) for candidate in candidate_envs]
            mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(device) for mask in mask_envs]

            with torch.no_grad():
                action_envs, a_idx_envs = [], []
                for i in range(train_parameters["num_envs"]):
                    # Compute action probabilities and select actions
                    pi, _ = ppo.policy_old(x=fea_tensor_envs[i],
                                           graph_pool=g_pool_step,
                                           padded_nei=None,
                                           adj=adj_tensor_envs[i],
                                           candidate=candidate_tensor_envs[i].unsqueeze(0),
                                           mask=mask_tensor_envs[i].unsqueeze(0))
                    action, a_idx = select_action(pi, candidate_envs[i], memories[i])
                    action_envs.append(action)
                    a_idx_envs.append(a_idx)

            # Reset states
            adj_envs, fea_envs, candidate_envs, mask_envs = [], [], [], []

            # Perform actions in each environment and record transitions
            for i in range(train_parameters["num_envs"]):
                memories[i].adj_mb.append(adj_tensor_envs[i])
                memories[i].fea_mb.append(fea_tensor_envs[i])
                memories[i].candidate_mb.append(candidate_tensor_envs[i])
                memories[i].mask_mb.append(mask_tensor_envs[i])
                memories[i].a_mb.append(a_idx_envs[i])

                adj, fea, reward, done, candidate, mask = envs[i].step(action_envs[i].item())
                adj_envs.append(adj)
                fea_envs.append(fea)
                candidate_envs.append(candidate)
                mask_envs.append(mask)
                ep_rewards[i] += reward
                memories[i].r_mb.append(reward)
                memories[i].done_mb.append(done)

            # Break if environments are done
            if envs[0].done():
                break

        # Post-episode reward adjustment
        for j in range(train_parameters["num_envs"]):
            ep_rewards[j] -= envs[j].posRewards

        # Update PPO policy and clear memories
        loss, v_loss = ppo.update(memories, n_job * n_machine, model_parameters["graph_pool_type"])
        for memory in memories:
            memory.clear_memory()

        # Logging
        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
        log.append([i_update, mean_rewards_all_env])

        # Periodic logging and validation
        if (i_update + 1) % 100 == 0:
            validation_result = -validate(vali_data, ppo.policy).mean()
            validation_log.append(validation_result)

            # Save training log
            log_file = f'./training_log/log_{n_job}_{n_machine}_{env_parameters["low"]}_{env_parameters["high"]}.txt'

            with open(log_file, 'w') as file_writing_obj:
                file_writing_obj.write(str(log))

            # Save validation log
            validation_file = f'./training_log/vali_{n_job}_{n_machine}_{env_parameters["low"]}_{env_parameters["high"]}.txt'
            with open(validation_file, 'w') as file_writing_obj1:
                file_writing_obj1.write(str(validation_log))

            # Save model if current validation result is the best so fa
            if validation_result < record:
                torch.save(ppo.policy.state_dict(), f'./saved_models/{n_job}_{n_machine}_{env_parameters["low"]}_{env_parameters["high"]}.pth')
                record = validation_result

            logging.info(
                f'Episode {i_update + 1}\t Last reward: {mean_rewards_all_env:.2f}\t Mean_Vloss: {v_loss:.8f}\t Validation quality: {validation_result:.2f}')


def main(param_file: str = PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    train_L2D(**parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train L2D")
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