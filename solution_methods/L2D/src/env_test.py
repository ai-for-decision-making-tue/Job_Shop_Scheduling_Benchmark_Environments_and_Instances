import sys
from pathlib import Path

import numpy as np

from scheduling_environment.jobShop import JobShop
from solution_methods.helper_functions import load_parameters

base_path = Path(__file__).resolve().parents[3]
sys.path.append(str(base_path))

param_file = str(base_path) + "/configs/L2D.toml"
parameters = load_parameters(param_file)
env_parameters = parameters["env_parameters"]
model_parameters = parameters["network_parameters"]
train_parameters = parameters["train_parameters"]
test_parameters = parameters["test_parameters"]


class NipsJSPEnv_test():
    def __init__(self, n_j: int, n_m: int):

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        # the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.JobShopModule: JobShop = None

    def reset(self, JSM_env: JobShop):
        self.JobShopModule = JSM_env
        self.JobShopModule.reset()

        self.step_count = 0

        # record action history
        self.partial_sol_sequeence = []
        self.posRewards = 0

        # initialize adj matrix
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.JSM_adj = self_as_nei + conj_nei_up_stream

        # initialize features
        self.JSM_LBs = np.zeros((len(self.JobShopModule.jobs), len(self.JobShopModule.machines)), dtype=np.single)
        for i in range(len(self.JobShopModule.jobs)):
            for j in range(len(self.JobShopModule.machines)):
                if j == 0:
                    self.JSM_LBs[i, j] = list(self.JobShopModule.jobs[i].operations[j].processing_times.values())[0]
                else:
                    self.JSM_LBs[i, j] = self.JSM_LBs[i, j-1] + list(self.JobShopModule.jobs[i].operations[j].processing_times.values())[0]

        self.JSM_max_endTime = self.JSM_LBs.max() if not env_parameters["init_quality_flag"] else 0
        self.JSM_finished_mark = np.zeros_like(self.JSM_LBs, dtype=np.single)
        self.initQuality = self.JSM_LBs.max() if not env_parameters["init_quality_flag"] else 0
        fea = np.concatenate((self.JSM_LBs.reshape(-1, 1) / env_parameters["et_normalize_coef"],
                              self.JSM_finished_mark.reshape(-1, 1)), axis=1)
        # initialize feasible omega
        self.JSM_omega = self.first_col.astype(np.int64)
        # initialize mask
        self.JSM_mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        return self.JSM_adj, fea, self.JSM_omega, self.JSM_mask

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    def step(self, action):
        # action is a int 0 - 224 for 15x15 for example
        # redundant action makes no effect

        ope_to_schedule = self.JobShopModule.get_operation(action)

        if len(ope_to_schedule.scheduling_information) == 0:
            self.partial_sol_sequeence.append(action)
            self.step_count += 1

            assigned_mach = list(ope_to_schedule.processing_times.keys())[0]
            process_time = list(ope_to_schedule.processing_times.values())[0]
            self.JobShopModule.schedule_operation_on_machine(ope_to_schedule, assigned_mach, process_time)
            job_id = ope_to_schedule.job_id
            ope_idx_in_job = ope_to_schedule.job.operations.index(ope_to_schedule)
            self.JSM_finished_mark[job_id, ope_idx_in_job] = 1

            self.JSM_adj[ope_to_schedule.operation_id] = 0
            self.JSM_adj[ope_to_schedule.operation_id, ope_to_schedule.operation_id] = 1
            if ope_idx_in_job != 0:
                self.JSM_adj[ope_to_schedule.operation_id, ope_to_schedule.operation_id-1] = 1
            machine = self.JobShopModule.get_machine(assigned_mach)
            ope_idx_in_machine = machine.scheduled_operations.index(ope_to_schedule)
            if ope_idx_in_machine > 0:
                prede_ope_id = machine.scheduled_operations[ope_idx_in_machine - 1].operation_id
                self.JSM_adj[ope_to_schedule.operation_id, prede_ope_id] = 1
            if ope_idx_in_machine < len(machine.scheduled_operations) - 1:
                succe_ope_id = machine.scheduled_operations[ope_idx_in_machine + 1].operation_id
                self.JSM_adj[succe_ope_id, ope_to_schedule.operation_id] = 1
                if ope_idx_in_machine > 0:
                    self.JSM_adj[succe_ope_id, prede_ope_id] = 0

            if action not in self.last_col:
                self.JSM_omega[job_id] += 1
            else:
                self.JSM_mask[job_id] = 1

            self.JSM_LBs[job_id, ope_idx_in_job] = ope_to_schedule.scheduling_information.get('end_time')
            for i in range(ope_idx_in_job + 1, len(ope_to_schedule.job.operations)):
                next_ope = ope_to_schedule.job.operations[i]
                pure_process_time = list(next_ope.processing_times.values())[0]
                self.JSM_LBs[job_id, i] = self.JSM_LBs[job_id, i-1] + pure_process_time

        # prepare for return
        feature_JSM = np.concatenate((self.JSM_LBs.reshape(-1, 1) / env_parameters["et_normalize_coef"],
                              self.JSM_finished_mark.reshape(-1, 1)), axis=1)
        reward_JSM = - (self.JSM_LBs.max() - self.JSM_max_endTime)

        if reward_JSM == 0:
            reward_JSM = env_parameters["rewardscale"]
            self.posRewards += reward_JSM

        self.JSM_max_endTime = self.JSM_LBs.max()

        return self.JSM_adj, feature_JSM, reward_JSM, self.done(), self.JSM_omega, self.JSM_mask