import sys
from pathlib import Path

import numpy as np

from solution_methods.helper_functions import load_parameters
from solution_methods.L2D.agent_utils import select_action
from solution_methods.L2D.JSSP_Env import SJSSP
from solution_methods.L2D.mb_agg import *
from solution_methods.L2D.PPO_model import PPO, Memory
from solution_methods.L2D.uniform_instance_gen import uni_instance_gen
from solution_methods.L2D.validation import validate

base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

param_file = str(base_path) + "/configs/L2D.toml"
parameters = load_parameters(param_file)
env_parameters = parameters["env_parameters"]
model_parameters = parameters["network_parameters"]
train_parameters = parameters["train_parameters"]
device = torch.device(env_parameters["device"])


def main():
    n_job = env_parameters["n_j"]
    n_machine = env_parameters["n_m"]

    envs = [SJSSP(n_j=n_job, n_m=n_machine) for _ in range(train_parameters["num_envs"])]

    data_generator = uni_instance_gen
    file_path = str(base_path) + '/solution_methods/L2D/generated_data/'
    dataLoaded = np.load(file_path+'generatedData' + str(env_parameters["n_j"]) + '_' + str(env_parameters["n_m"])
                         + '_Seed' + str(env_parameters["np_seed_validation"]) + '.npy')
    vali_data = []
    for i in range(dataLoaded.shape[0]):
        vali_data.append((dataLoaded[i][0], dataLoaded[i][1]))

    torch.manual_seed(env_parameters["torch_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(env_parameters["torch_seed"])
    np.random.seed(env_parameters["np_seed_train"])

    memories = [Memory() for _ in range(train_parameters["num_envs"])]

    ppo = PPO(train_parameters["lr"], train_parameters["gamma"], train_parameters["k_epochs"],
              train_parameters["eps_clip"],
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
    g_pool_step = g_pool_cal(graph_pool_type=model_parameters["graph_pool_type"],
                             batch_size=torch.Size([1, n_job * n_machine, n_job * n_machine]),
                             n_nodes=n_job * n_machine, device=device)
    # training loop
    log = []
    validation_log = []
    record = 100000
    for i_update in range(train_parameters["max_updates"]):

        ep_rewards = [0 for _ in range(train_parameters["num_envs"])]
        adj_envs = []
        fea_envs = []
        candidate_envs = []
        mask_envs = []

        for i, env in enumerate(envs):
            adj, fea, candidate, mask = env.reset(data_generator(n_j=n_job, n_m=n_machine, low=env_parameters['low'],
                                                                 high=env_parameters["high"]))
            adj_envs.append(adj)
            fea_envs.append(fea)
            candidate_envs.append(candidate)
            mask_envs.append(mask)
            ep_rewards[i] = - env.initQuality

        # rollout the env
        while True:
            fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(device) for fea in fea_envs]
            adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(device).to_sparse() for adj in adj_envs]
            candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(device) for candidate in candidate_envs]
            mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(device) for mask in mask_envs]

            with torch.no_grad():
                action_envs = []
                a_idx_envs = []
                for i in range(train_parameters["num_envs"]):

                    pi, xxx = ppo.policy_old(x=fea_tensor_envs[i],
                                             graph_pool=g_pool_step,
                                             padded_nei=None,
                                             adj=adj_tensor_envs[i],
                                             candidate=candidate_tensor_envs[i].unsqueeze(0),
                                             mask=mask_tensor_envs[i].unsqueeze(0))
                    # print('old', pi.shape,xxx.shape)
                    action, a_idx = select_action(pi, candidate_envs[i], memories[i])
                    action_envs.append(action)
                    a_idx_envs.append(a_idx)

            adj_envs = []
            fea_envs = []
            candidate_envs = []
            mask_envs = []
            # Saving episode data
            for i in range(train_parameters["num_envs"]):
                memories[i].adj_mb.append(adj_tensor_envs[i])
                memories[i].fea_mb.append(fea_tensor_envs[i])
                memories[i].candidate_mb.append(candidate_tensor_envs[i])
                memories[i].mask_mb.append(mask_tensor_envs[i])
                memories[i].a_mb.append(a_idx_envs[i])

                adj, fea, reward, done, candidate, mask = envs[i].step(action_envs[i].item())
                # print('returned raw feature shape', fea)
                adj_envs.append(adj)
                fea_envs.append(fea)
                candidate_envs.append(candidate)
                mask_envs.append(mask)
                ep_rewards[i] += reward
                memories[i].r_mb.append(reward)
                memories[i].done_mb.append(done)
            if envs[0].done():
                break
        for j in range(train_parameters["num_envs"]):
            ep_rewards[j] -= envs[j].posRewards

        loss, v_loss = ppo.update(memories, n_job * n_machine, model_parameters["graph_pool_type"])
        for memory in memories:
            memory.clear_memory()
        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
        log.append([i_update, mean_rewards_all_env])
        if (i_update + 1) % 100 == 0:
            file_writing_obj = open(
                './' + 'log_' + str(n_job) + '_' + str(n_machine) + '_' + str(env_parameters["low"]) + '_' + str(
                    env_parameters["high"]) + '.txt', 'w')
            file_writing_obj.write(str(log))

        # log results
        print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}'.format(
            i_update + 1, mean_rewards_all_env, v_loss))

        # validate and save use mean performance
        if (i_update + 1) % 100 == 0:
            vali_result = - validate(vali_data, ppo.policy).mean()
            validation_log.append(vali_result)
            if vali_result < record:
                torch.save(ppo.policy.state_dict(), './{}.pth'.format(
                    str(n_job) + '_' + str(n_machine) + '_' + str(env_parameters["low"]) + '_' + str(env_parameters["high"])))
                record = vali_result
            print('The validation quality is:', vali_result)
            file_writing_obj1 = open(
                './' + 'vali_' + str(n_job) + '_' + str(n_machine) + '_' + str(env_parameters["low"]) + '_' + str(
                    env_parameters["high"]) + '.txt', 'w')
            file_writing_obj1.write(str(validation_log))


if __name__ == '__main__':
    main()
