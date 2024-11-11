import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from scheduling_environment.jobShop import JobShop
from solution_methods.FJSP_DRL.src.load_data import load_feats_from_sim

# Add the base path to the Python module search path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))


@dataclass
class EnvState:
    '''
    Class for the state of the environment
    '''
    # static
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None

    # dynamic
    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch: torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None

    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, proc_times_batch, ope_ma_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch

        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time


def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    """
    Convert job features into operation features (such as dimension)
    """
    return feat_job_batch.gather(1, opes_appertain_batch)


class FJSPEnv_test():
    def __init__(self, JobShop_module, test_parameters):
        # static
        self.batch_size = 1
        self.num_jobs = JobShop_module.nr_of_jobs  # Number of jobs
        self.num_mas = JobShop_module.nr_of_machines  # Number of machines
        self.device = test_parameters["device"]  # Computing device for PyTorch

        self.JSP_instance: list[JobShop] = JobShop_module

        # load instance
        num_data = 8  # The amount of data extracted from instance
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0
        self.num_opes = max(self.num_jobs, len(self.JSP_instance.operations))

        # Extract features from each JobShop module
        raw_features = load_feats_from_sim(self.JSP_instance, self.num_mas, self.num_opes)
        # print(raw_features[0].shape)
        for j in range(num_data):
            tensors[j].append(raw_features[j].to(self.device))

        # dynamic feats
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()

        # static feats
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
        self.JSM_time = torch.zeros(self.batch_size)  # Current time of the environment
        self.N = torch.zeros(self.batch_size).int()  # Count scheduled operations
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)

        # Generate raw feature vectors
        ope_feat_dim = 6
        ma_feat_dim = 3
        num_sample = 1
        feat_opes_batch = torch.zeros(size=(num_sample, ope_feat_dim, self.num_opes))
        feat_mas_batch = torch.zeros(size=(num_sample, ma_feat_dim, self.num_mas))

        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1), self.cal_cumul_adj_batch).squeeze()
        end_time_batch = (feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        self.JSM_feat_opes_batch = feat_opes_batch
        self.JSM_feat_mas_batch = feat_mas_batch

        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for jobs in process
        self.JSM_mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                     fill_value=False)
        # shape: (batch_size, num_jobs), True for completed jobs
        self.JSM_mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                    fill_value=False)
        # shape: (batch_size, num_mas), True for machines in process
        self.JSM_mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                    fill_value=False)

        self.makespan_batch = torch.max(self.JSM_feat_opes_batch[:, 4, :], dim=1)[0]  # shape: (batch_size)
        self.done_batch = self.JSM_mask_job_finish_batch.all(dim=1)  # shape: (batch_size)

        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.JSM_feat_opes_batch, feat_mas_batch=self.JSM_feat_mas_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
                              mask_job_procing_batch=self.JSM_mask_job_procing_batch,
                              mask_job_finish_batch=self.JSM_mask_job_finish_batch,
                              mask_ma_procing_batch=self.JSM_mask_ma_procing_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              time_batch=self.JSM_time, nums_opes_batch=self.nums_opes)

        # Save initial data for reset
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.JSM_feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.JSM_feat_mas_batch)
        self.old_state = copy.deepcopy(self.state)

    def step(self, actions):
        """
        Environment transition function, based on JobShop module
        """
        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]
        self.N += 1

        ope_idx = opes.item()
        mac_idx = mas.item()
        env = self.JSP_instance
        operation = env.operations[ope_idx]
        duration = operation.processing_times[mac_idx]
        env.schedule_operation_on_machine(operation, mac_idx, duration)
        env.get_job(operation.job_id).scheduled_operations.append(operation)

        # Removed unselected O-M arcs of the scheduled operations
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]
        self.JSM_feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack(
            (torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             proc_times), dim=1)

        # Update 'Number of unscheduled operations in the job' - use JobShop

        job_idx = jobs.item()
        unscheduled_opes = 0
        for each_ope in self.JSP_instance.get_job(job_idx).operations:
            if each_ope.scheduling_information.__len__() == 0:
                unscheduled_opes += 1
        start_ope_idx = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope_idx = self.end_ope_biases_batch[self.batch_idxes, jobs]
        self.JSM_feat_opes_batch[self.batch_idxes, 3, start_ope_idx:end_ope_idx + 1] = unscheduled_opes

        # Update 'Start time' and 'Job completion time' - use JobShop
        self.JSM_feat_opes_batch[self.batch_idxes, 5, opes] = self.JSM_time
        for each_ope in self.JSP_instance.operations:
            if each_ope.scheduling_information.__len__() == 0:
                if each_ope.predecessors.__len__() == 0:
                    est_start_time = self.JSM_feat_opes_batch[self.batch_idxes, 5, each_ope.operation_id]
                else:
                    if each_ope.predecessors[0].scheduling_information.__len__() == 0:
                        est_start_time = self.JSM_feat_opes_batch[self.batch_idxes, 5, each_ope.predecessors[0].operation_id] + self.JSM_feat_opes_batch[self.batch_idxes, 2, each_ope.predecessors[0].operation_id]
                    else:
                        est_start_time = each_ope.predecessors[0].scheduling_information['start_time'] + each_ope.predecessors[0].scheduling_information['processing_time']
            else:
                est_start_time = each_ope.scheduling_information['start_time']
            self.JSM_feat_opes_batch[self.batch_idxes, 5, each_ope.operation_id] = est_start_time

        for each_job in self.JSP_instance.jobs:
            est_end_times = [(self.JSM_feat_opes_batch[self.batch_idxes, 5, ope_in_job.operation_id] + self.JSM_feat_opes_batch[self.batch_idxes, 2, ope_in_job.operation_id]) for ope_in_job in each_job.operations]
            job_end_time = max(est_end_times)
            for ope_of_job in each_job.operations:
                self.JSM_feat_opes_batch[self.batch_idxes, 4, ope_of_job.operation_id] = job_end_time

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time - using JobShop Module
        this_env = self.JSP_instance
        cur_times = []
        for each_job in this_env.jobs:
            if len(each_job.scheduled_operations) == len(each_job.operations):
                continue
            next_ope = each_job.operations[len(each_job.scheduled_operations)]
            latest_ope_end_time = next_ope.finishing_time_predecessors
            for each_mach_id in next_ope.optional_machines_id:
                # if schedule the next operation of this job on an optional machine, the earlist start time
                # operation available: predecessor operation end
                # machine available: last assigned operation end
                cur_times.append(max(latest_ope_end_time, this_env.get_machine(each_mach_id).next_available_time, self.JSM_time))
        self.JSM_time = min(cur_times, default=self.JSM_time)

        # Update feature vectors of machines - using JobShop module
        self.JSM_feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :], dim=1).float()
        for each_mach in self.JSP_instance.machines:
            workload = sum([ope_on_mach.scheduled_duration for ope_on_mach in each_mach.scheduled_operations])
            cur_time = self.JSM_time
            workload = min(cur_time, workload)
            self.JSM_feat_mas_batch[self.batch_idxes, 2, each_mach.machine_id] = workload / (cur_time + 1e-9)
            self.JSM_feat_mas_batch[self.batch_idxes, 1, each_mach.machine_id] = each_mach.next_available_time

        # Update other variable according to actions - using JobShop module
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        JSM_mask_jp_list = []
        JSM_mask_jf_list = []
        JSM_mask_mp_list = []
        # JSM_mask_jp_list.append([True if this_job.next_ope_earliest_begin_time > self.JSM_time else False
        #                          for this_job in self.JSP_instance.jobs])

        JSM_mask_jp_list.append([True if max([operation.scheduled_end_time for operation in this_job.scheduled_operations], default=0) > self.JSM_time else False
                                 for this_job in self.JSP_instance.jobs])

        JSM_mask_jf_list.append([True if this_job.operations.__len__() == this_job.scheduled_operations.__len__() else False for this_job in self.JSP_instance.jobs])
        JSM_mask_mp_list.append([True if this_mach.next_available_time > self.JSM_time else False for this_mach in self.JSP_instance.machines])
        self.JSM_mask_job_procing_batch = torch.tensor(JSM_mask_jp_list, dtype=torch.bool)
        self.JSM_mask_job_finish_batch = torch.tensor(JSM_mask_jf_list, dtype=torch.bool)
        self.JSM_mask_ma_procing_batch = torch.tensor(JSM_mask_mp_list, dtype=torch.bool)

        self.done_batch = self.JSM_mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

        max_makespan = torch.max(self.JSM_feat_opes_batch[:, 4, :], dim=1)[0]
        self.reward_batch = self.makespan_batch - max_makespan
        self.makespan_batch = max_makespan

        # Update the vector for uncompleted instances
        mask_finish = (self.N + 1) <= self.nums_opes
        if ~(mask_finish.all()):
            self.batch_idxes = torch.arange(self.batch_size)[mask_finish]

        # Update state of the environment
        self.state.update(self.batch_idxes, self.JSM_feat_opes_batch, self.JSM_feat_mas_batch, self.proc_times_batch,
                          self.ope_ma_adj_batch, self.JSM_mask_job_procing_batch, self.JSM_mask_job_finish_batch,
                          self.JSM_mask_ma_procing_batch, self.ope_step_batch, self.JSM_time)
        return self.state, self.reward_batch, self.done_batch

    def reset(self):
        """
        Reset the environment to its initial state
        """
        for i in range(self.batch_size):
            self.JSP_instance[i].reset()

        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.JSM_feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.JSM_feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.state = copy.deepcopy(self.old_state)

        self.batch_idxes = torch.arange(self.batch_size)

        self.JSM_time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)

        self.JSM_mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                     fill_value=False)
        self.JSM_mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                    fill_value=False)
        self.JSM_mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                    fill_value=False)
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.JSM_feat_opes_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.JSM_mask_job_finish_batch.all(dim=1)

        return self.state
