from solution_methods.DANIEL.network.main_model import DANIEL
from solution_methods.DANIEL.src.common_utils import eval_actions
import torch.nn as nn
import torch
from copy import deepcopy
import numpy as np


class Memory:
    def __init__(self, gamma, gae_lambda):
        """
            the memory used for collect trajectories for PPO training
        :param gamma: discount factor
        :param gae_lambda: GAE parameter for PPO algorithm
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # input variables of DANIEL
        self.fea_j_seq = []  # [N, tensor[sz_b, N, 8]]
        self.op_mask_seq = []  # [N, tensor[sz_b, N, 3]]
        self.fea_m_seq = []  # [N, tensor[sz_b, M, 6]]
        self.mch_mask_seq = []  # [N, tensor[sz_b, M, M]]
        self.dynamic_pair_mask_seq = []  # [N, tensor[sz_b, J, M]]
        self.comp_idx_seq = []  # [N, tensor[sz_b, M, M, J]]
        self.candidate_seq = []  # [N, tensor[sz_b, J]]
        self.fea_pairs_seq = []  # [N, tensor[sz_b, J]]

        # other variables
        self.action_seq = []  # action index with shape [N, tensor[sz_b]]
        self.reward_seq = []  # reward value with shape [N, tensor[sz_b]]
        self.val_seq = []  # state value with shape [N, tensor[sz_b]]
        self.done_seq = []  # done flag with shape [N, tensor[sz_b]]
        self.log_probs = []  # log(p_{\theta_old}(a_t|s_t)) with shape [N, tensor[sz_b]]

    def clear_memory(self):
        self.clear_state()
        del self.action_seq[:]
        del self.reward_seq[:]
        del self.val_seq[:]
        del self.done_seq[:]
        del self.log_probs[:]

    def clear_state(self):
        del self.fea_j_seq[:]
        del self.op_mask_seq[:]
        del self.fea_m_seq[:]
        del self.mch_mask_seq[:]
        del self.dynamic_pair_mask_seq[:]
        del self.comp_idx_seq[:]
        del self.candidate_seq[:]
        del self.fea_pairs_seq[:]

    def push(self, state):
        """
            push a state into the memory
        :param state: the MDP state
        :return:
        """
        self.fea_j_seq.append(state.fea_j_tensor)
        self.op_mask_seq.append(state.op_mask_tensor)
        self.fea_m_seq.append(state.fea_m_tensor)
        self.mch_mask_seq.append(state.mch_mask_tensor)
        self.dynamic_pair_mask_seq.append(state.dynamic_pair_mask_tensor)
        self.comp_idx_seq.append(state.comp_idx_tensor)
        self.candidate_seq.append(state.candidate_tensor)
        self.fea_pairs_seq.append(state.fea_pairs_tensor)

    def transpose_data(self):
        """
            transpose the first and second dimension of collected variables
        """
        # 14
        t_Fea_j_seq = torch.stack(self.fea_j_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_op_mask_seq = torch.stack(self.op_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_Fea_m_seq = torch.stack(self.fea_m_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_mch_mask_seq = torch.stack(self.mch_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_dynamicMask_seq = torch.stack(self.dynamic_pair_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_Compete_m_seq = torch.stack(self.comp_idx_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_candidate_seq = torch.stack(self.candidate_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_pairMessage_seq = torch.stack(self.fea_pairs_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_action_seq = torch.stack(self.action_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_reward_seq = torch.stack(self.reward_seq, dim=0).transpose(0, 1).flatten(0, 1)
        self.t_old_val_seq = torch.stack(self.val_seq, dim=0).transpose(0, 1)
        t_val_seq = self.t_old_val_seq.flatten(0, 1)
        t_done_seq = torch.stack(self.done_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_logprobs_seq = torch.stack(self.log_probs, dim=0).transpose(0, 1).flatten(0, 1)

        return t_Fea_j_seq, t_op_mask_seq, t_Fea_m_seq, t_mch_mask_seq, t_dynamicMask_seq, \
               t_Compete_m_seq, t_candidate_seq, t_pairMessage_seq, \
               t_action_seq, t_reward_seq, t_val_seq, t_done_seq, t_logprobs_seq

    def get_gae_advantages(self):
        """
            Compute the generalized advantage estimates
        :return: advantage sequences, state value sequence
        """

        reward_arr = torch.stack(self.reward_seq, dim=0)
        values = self.t_old_val_seq.transpose(0, 1)
        len_trajectory, len_envs = reward_arr.shape

        advantage = torch.zeros(len_envs, device=values.device)
        advantage_seq = []
        for i in reversed(range(len_trajectory)):

            if i == len_trajectory - 1:
                delta_t = reward_arr[i] - values[i]
            else:
                delta_t = reward_arr[i] + self.gamma * values[i + 1] - values[i]
            advantage = delta_t + self.gamma * self.gae_lambda * advantage
            advantage_seq.insert(0, advantage)

        # [sz_b, N]
        t_advantage_seq = torch.stack(advantage_seq, dim=0).transpose(0, 1).to(torch.float32)

        # [sz_b, N]
        v_target_seq = (t_advantage_seq + self.t_old_val_seq).flatten(0, 1)

        # normalization
        t_advantage_seq = (t_advantage_seq - t_advantage_seq.mean(dim=1, keepdim=True)) \
                          / (t_advantage_seq.std(dim=1, keepdim=True) + 1e-8)

        return t_advantage_seq.flatten(0, 1), v_target_seq


class PPO:
    def __init__(self, config):
        """
            The implementation of PPO algorithm
        :param config: a package of parameters
        """
        self.lr = config["PPO_Algorithm"]["lr"]
        self.gamma = config["PPO_Algorithm"]["gamma"]
        self.gae_lambda = config["PPO_Algorithm"]["gae_lambda"]
        self.eps_clip = config["PPO_Algorithm"]["eps_clip"]
        self.k_epochs = config["PPO_Algorithm"]["k_epochs"]
        self.tau = config["PPO_Algorithm"]["tau"]

        self.ploss_coef = config["PPO_Algorithm"]["ploss_coef"]
        self.vloss_coef = config["PPO_Algorithm"]["vloss_coef"]
        self.entloss_coef = config["PPO_Algorithm"]["entloss_coef"]
        self.minibatch_size = config["training"]["minibatch_size"]

        self.policy = DANIEL(config)
        self.policy_old = deepcopy(self.policy)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.V_loss_2 = nn.MSELoss()
        self.device = torch.device(config["device"]["name"])

    def update(self, memory):
        '''
        :param memory: data used for PPO training
        :return: total_loss and critic_loss
        '''

        t_data = memory.transpose_data()

        t_advantage_seq, v_target_seq = memory.get_gae_advantages()

        full_batch_size = len(t_data[-1])
        num_batch = np.ceil(full_batch_size / self.minibatch_size)

        loss_epochs = 0
        v_loss_epochs = 0

        for _ in range(self.k_epochs):

            # Split into multiple batches of updates due to memory limitations
            for i in range(int(num_batch)):
                if i + 1 < num_batch:
                    start_idx = i * self.minibatch_size
                    end_idx = (i + 1) * self.minibatch_size
                else:
                    # the last batch
                    start_idx = i * self.minibatch_size
                    end_idx = full_batch_size

                pis, vals = self.policy(fea_j=t_data[0][start_idx:end_idx],
                                        op_mask=t_data[1][start_idx:end_idx],
                                        candidate=t_data[6][start_idx:end_idx],
                                        fea_m=t_data[2][start_idx:end_idx],
                                        mch_mask=t_data[3][start_idx:end_idx],
                                        comp_idx=t_data[5][start_idx:end_idx],
                                        dynamic_pair_mask=t_data[4][start_idx:end_idx],
                                        fea_pairs=t_data[7][start_idx:end_idx])

                action_batch = t_data[8][start_idx: end_idx]
                logprobs, ent_loss = eval_actions(pis, action_batch)
                ratios = torch.exp(logprobs - t_data[12][start_idx: end_idx].detach())

                advantages = t_advantage_seq[start_idx: end_idx]
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                v_loss = self.V_loss_2(vals.squeeze(1), v_target_seq[start_idx: end_idx])
                p_loss = - torch.min(surr1, surr2)
                ent_loss = - ent_loss.clone()
                loss = self.vloss_coef * v_loss + self.ploss_coef * p_loss + self.entloss_coef * ent_loss

                self.optimizer.zero_grad()
                loss_epochs += loss.mean().detach()
                v_loss_epochs += v_loss.mean().detach()
                loss.mean().backward()
                self.optimizer.step()
        # soft update
        for policy_old_params, policy_params in zip(self.policy_old.parameters(), self.policy.parameters()):
            policy_old_params.data.copy_(self.tau * policy_old_params.data + (1 - self.tau) * policy_params.data)

        return loss_epochs.item() / self.k_epochs, v_loss_epochs.item() / self.k_epochs


def PPO_initialize(config):
    ppo = PPO(config=config)
    return ppo
