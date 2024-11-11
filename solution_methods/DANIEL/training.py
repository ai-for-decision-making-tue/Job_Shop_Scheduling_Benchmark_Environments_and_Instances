import argparse
import os
import random
import sys
import time
from copy import deepcopy
import logging
from pathlib import Path
import numpy as np
import torch

from tqdm import tqdm

from solution_methods.DANIEL.src.common_utils import greedy_select_action, sample_action, setup_seed, strToSuffix
from solution_methods.DANIEL.src.data_utils import CaseGenerator, SD2_instance_generator, load_data_from_files
from solution_methods.DANIEL.src.fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from solution_methods.DANIEL.src.fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from solution_methods.DANIEL.network.PPO import Memory, PPO_initialize
from solution_methods.helper_functions import load_parameters, initialize_device

base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

PARAM_FILE = str(base_path) + "/configs/DANIEL.toml"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


class Trainer:
    def __init__(self, config, device):

        self.n_j = config["env"]["n_j"]
        self.n_m = config["env"]["n_m"]
        self.low = config["env"]["low"]
        self.high = config["env"]["high"]
        self.op_per_job_min = int(0.8 * self.n_m)
        self.op_per_job_max = int(1.2 * self.n_m)
        self.data_source = config["data"]["source"]
        self.config = config
        self.max_updates = config["PPO_Algorithm"]["max_updates"]
        self.reset_env_timestep = config["training"]["reset_env_timestep"]
        self.validate_timestep = config["training"]["validate_timestep"]
        self.num_envs = config["PPO_Algorithm"]["num_envs"]
        self.device = device

        if not os.path.exists(f"./save/{self.data_source}"):
            os.makedirs(f"./save/{self.data_source}")
        if not os.path.exists(
            f"./train_log/{self.data_source}"
        ):
            os.makedirs(f"./train_log/{self.data_source}")

        if device.type == "cuda":
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        if self.data_source == "SD1":
            self.data_name = f"{self.n_j}x{self.n_m}"
        elif self.data_source == "SD2":
            self.data_name = (
                f'{self.n_j}x{self.n_m}{strToSuffix(config["data"]["suffix"])}'
            )

        self.vali_data_path = f"./data/data_train_vali/{self.data_source}/{self.data_name}"
        self.test_data_path = (
            f"./data/{self.data_source}/{self.data_name}"
        )
        self.model_name = f'{self.data_name}{strToSuffix(config["model"]["suffix"])}'

        # seed
        self.seed_train = config["seed"]["seed_train"]
        setup_seed(self.seed_train)

        self.env = FJSPEnvForSameOpNums(self.n_j, self.n_m, device)
        self.test_data = load_data_from_files(self.test_data_path)
        # validation data set
        vali_data = load_data_from_files(self.vali_data_path)

        if self.data_source == "SD1":
            self.vali_env = FJSPEnvForVariousOpNums(self.n_j, self.n_m, device)
        elif self.data_source == "SD2":
            self.vali_env = FJSPEnvForSameOpNums(self.n_j, self.n_m, device)

        self.vali_env.set_initial_data(vali_data[0], vali_data[1])

        self.ppo = PPO_initialize(config)
        self.memory = Memory(
            gamma=config["PPO_Algorithm"]["gamma"],
            gae_lambda=config["PPO_Algorithm"]["gae_lambda"],
        )

    def train(self):
        """
        train the model following the config
        """
        setup_seed(self.seed_train)
        self.log = []
        self.validation_log = []
        self.record = float("inf")

        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")

        self.train_st = time.time()

        for i_update in tqdm(
            range(self.max_updates), file=sys.stdout, desc="progress", colour="blue"
        ):
            ep_st = time.time()

            # resampling the training data
            if i_update % self.reset_env_timestep == 0:
                dataset_job_length, dataset_op_pt = self.sample_training_instances()
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
            else:
                state = self.env.reset()

            ep_rewards = -deepcopy(self.env.init_quality)

            while True:

                # state store
                self.memory.push(state)
                with torch.no_grad():

                    pi_envs, vals_envs = self.ppo.policy_old(
                        fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                        op_mask=state.op_mask_tensor,  # [sz_b, N, N]
                        candidate=state.candidate_tensor,  # [sz_b, J]
                        fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                        mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                        comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                        fea_pairs=state.fea_pairs_tensor,
                    )  # [sz_b, J, M]

                # sample the action
                action_envs, action_logprob_envs = sample_action(pi_envs)

                # state transition
                state, reward, done = self.env.step(actions=action_envs.cpu().numpy())
                ep_rewards += reward
                reward = torch.from_numpy(reward).to(self.device)

                # collect the transition
                self.memory.done_seq.append(torch.from_numpy(done).to(self.device))
                self.memory.reward_seq.append(reward)
                self.memory.action_seq.append(action_envs)
                self.memory.log_probs.append(action_logprob_envs)
                self.memory.val_seq.append(vals_envs.squeeze(1))

                if done.all():
                    break

            loss, v_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()

            mean_rewards_all_env = np.mean(ep_rewards)
            mean_makespan_all_env = np.mean(self.env.current_makespan)

            # save the mean rewards of all instances in current training data
            self.log.append([i_update, mean_rewards_all_env])

            # validate the trained model
            if (i_update + 1) % self.validate_timestep == 0:
                if self.data_source == "SD1":
                    vali_result = self.validate_envs_with_various_op_nums().mean()
                else:
                    vali_result = self.validate_envs_with_same_op_nums().mean()

                if vali_result < self.record:
                    self.save_model()
                    self.record = vali_result

                self.validation_log.append(vali_result)
                self.save_validation_log()
                tqdm.write(
                    f"The validation quality is: {vali_result} (best : {self.record})"
                )

            ep_et = time.time()

            # print the reward, makespan, loss and training time of the current episode
            tqdm.write(
                "Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}".format(
                    i_update + 1,
                    mean_rewards_all_env,
                    mean_makespan_all_env,
                    loss,
                    ep_et - ep_st,
                )
            )

        self.train_et = time.time()

        # log results
        self.save_training_log()

    def save_training_log(self):
        """
        save reward data & validation makespan data (during training) and the entire training time
        """
        file_writing_obj = open(
            f"./train_log/{self.data_source}/"
            + "reward_"
            + self.model_name
            + ".txt",
            "w",
        )
        file_writing_obj.write(str(self.log))

        file_writing_obj1 = open(
            f"./train_log/{self.data_source}/"
            + "valiquality_"
            + self.model_name
            + ".txt",
            "w",
        )
        file_writing_obj1.write(str(self.validation_log))

    def save_validation_log(self):
        """
        save the results of validation
        """
        file_writing_obj1 = open(
            f"./train_log/{self.data_source}/"
            + "valiquality_"
            + self.model_name
            + ".txt",
            "w",
        )
        file_writing_obj1.write(str(self.validation_log))

    def sample_training_instances(self):
        """
            sample training instances following the config,
            the sampling process of SD1 data is imported from "songwenas12/fjsp-drl"
        :return: new training instances
        """
        prepare_JobLength = [
            random.randint(self.op_per_job_min, self.op_per_job_max)
            for _ in range(self.n_j)
        ]
        dataset_JobLength = []
        dataset_OpPT = []
        for i in range(self.num_envs):
            if self.data_source == "SD1":
                case = CaseGenerator(
                    self.n_j,
                    self.n_m,
                    self.op_per_job_min,
                    self.op_per_job_max,
                    nums_ope=prepare_JobLength,
                    path="./test",
                    flag_doc=False,
                )
                JobLength, OpPT, _ = case.get_case(i)

            else:
                JobLength, OpPT, _ = SD2_instance_generator(config=self.config)
            dataset_JobLength.append(JobLength)
            dataset_OpPT.append(OpPT)

        return dataset_JobLength, dataset_OpPT

    def validate_envs_with_same_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have the same number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                pi, _ = self.ppo.policy(
                    fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                    op_mask=state.op_mask_tensor,
                    candidate=state.candidate_tensor,  # [sz_b, J]
                    fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                    mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                    comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                    fea_pairs=state.fea_pairs_tensor,
                )  # [sz_b, J, M]

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        return self.vali_env.current_makespan

    def validate_envs_with_various_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have various number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                batch_idx = ~torch.from_numpy(self.vali_env.done_flag)
                pi, _ = self.ppo.policy(
                    fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
                    op_mask=state.op_mask_tensor[batch_idx],
                    candidate=state.candidate_tensor[batch_idx],  # [sz_b, J]
                    fea_m=state.fea_m_tensor[batch_idx],  # [sz_b, M, 6]
                    mch_mask=state.mch_mask_tensor[batch_idx],  # [sz_b, M, M]
                    comp_idx=state.comp_idx_tensor[batch_idx],  # [sz_b, M, M, J]
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor[
                        batch_idx
                    ],  # [sz_b, J, M]
                    fea_pairs=state.fea_pairs_tensor[batch_idx],
                )  # [sz_b, J, M]

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        return self.vali_env.current_makespan

    def save_model(self):
        """
        save the model
        """
        torch.save(
            self.ppo.policy.state_dict(),
            f"./save/{self.data_source}"
            f"/{self.model_name}.pth",
        )

    def load_model(self):
        """
        load the trained model
        """
        model_path = (
            f"./save/{self.data_source}/{self.model_name}.pth"
        )
        self.ppo.policy.load_state_dict(torch.load(model_path, map_location="cuda"))


def main(param_file: str = PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return
    device = initialize_device(parameters, method="DANIEL")
    trainer = Trainer(parameters, device)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train DANIEL")
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
