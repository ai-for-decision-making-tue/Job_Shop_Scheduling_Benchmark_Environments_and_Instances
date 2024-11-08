import sys
from pathlib import Path

import numpy as np
import torch

from scheduling_environment.jobShop import JobShop
from scheduling_environment.operation import Operation
from solution_methods.helper_functions import load_job_shop_env

# Add the base path to the Python module search path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))


def load_feats_from_case(lines, num_mas, num_opes):
    """
    Load the local FJSP instance.
    """
    flag = 0
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))
    matrix_pre_proc = torch.full(size=(num_opes, num_opes), dtype=torch.bool, fill_value=False)
    matrix_cal_cumul = torch.zeros(size=(num_opes, num_opes)).int()
    nums_ope = []  # A list of the number of operations for each job
    opes_appertain = np.array([])
    num_ope_biases = []  # The id of the first operation of each job
    # Parse data line by line
    for line in lines:
        # first line
        if flag == 0:
            flag += 1
        # last line
        elif line == "\n":
            break
        # other
        else:
            num_ope_bias = int(sum(nums_ope))  # The id of the first operation of this job
            num_ope_biases.append(num_ope_bias)
            # Detect information of this job and return the number of operations
            num_ope = edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul)
            nums_ope.append(num_ope)
            # nums_option = np.concatenate((nums_option, num_option))
            opes_appertain = np.concatenate((opes_appertain, np.ones(num_ope) * (flag - 1)))
            flag += 1
    matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)
    # Fill zero if the operations are insufficient (for parallel computation)
    opes_appertain = np.concatenate((opes_appertain, np.zeros(num_opes - opes_appertain.size)))
    return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc.t(), \
           torch.tensor(opes_appertain).int(), torch.tensor(num_ope_biases).int(), \
           torch.tensor(nums_ope).int(), matrix_cal_cumul


def load_fjs(path, num_mas, num_opes, num_jobs):
    """
    Load the local FJSP instance.
    """
    jobShopEnv = load_job_shop_env(path, from_absolute_path=True)
    drl_tensors = load_feats_from_sim(jobShopEnv, num_mas, num_opes)

    return drl_tensors, jobShopEnv


def load_feats_from_sim(jobShopEnv: JobShop, num_mas, num_opes):
    """convert scheduling_environment environment to DRL environment"""
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))
    matrix_ope_ma_adj = torch.zeros(size=(num_opes, num_mas)).int()
    matrix_pre_proc = torch.full(size=(num_opes, num_opes), dtype=torch.bool, fill_value=False)
    matrix_cal_cumul = torch.zeros(size=(num_opes, num_opes)).int()
    opes_appertain = torch.zeros(size=(num_opes,)).int()
    num_ope_biases = torch.zeros(size=(jobShopEnv.nr_of_jobs,)).int()
    nums_ope = torch.zeros(size=(jobShopEnv.nr_of_jobs,)).int()

    for job in jobShopEnv.jobs:
        num_ope_biases[job.job_id] = job.operations[0].operation_id
        nums_ope[job.job_id] = len(job.operations)

        nr_remaining_ops = len(job.operations)
        for operation in job.operations:
            nr_remaining_ops -= 1
            for op in range(1, nr_remaining_ops + 1):
                matrix_cal_cumul[operation.operation_id][operation.operation_id + op] = 1
    for operation in jobShopEnv.operations:
        opes_appertain[operation.operation_id] = operation.job_id
        for machine_id, duration in operation.processing_times.items():
            matrix_proc_time[operation.operation_id][machine_id] = duration
            matrix_ope_ma_adj[operation.operation_id][machine_id] = 1
            for predecessor in operation.predecessors:
                predecessor: Operation
                matrix_pre_proc[predecessor.operation_id][operation.operation_id] = True

    return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc.t(), opes_appertain, num_ope_biases, \
           nums_ope, matrix_cal_cumul


def nums_detec(lines):
    """
    Count the number of jobs, machines and operations
    """
    num_opes = 0
    for i in range(1, len(lines)):
        num_opes += int(lines[i].strip().split()[0]) if lines[i] != "\n" else 0
    line_split = lines[0].strip().split()
    num_jobs = int(line_split[0])
    num_mas = int(line_split[1])
    return num_jobs, num_mas, num_opes


def edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul):
    """
    Detect information of a job
    """
    line_split = line.split()
    flag = 0
    flag_time = 0
    flag_new_ope = 1
    idx_ope = -1
    num_ope = 0  # Store the number of operations of this job
    num_option = np.array([])  # Store the number of processable machines for each operation of this job
    mac = 0
    for i in line_split:
        x = int(i)
        # The first number indicates the number of operations of this job
        if flag == 0:
            num_ope = x
            flag += 1
        # new operation detected
        elif flag == flag_new_ope:
            idx_ope += 1
            flag_new_ope += x * 2 + 1
            num_option = np.append(num_option, x)
            if idx_ope != num_ope - 1:
                matrix_pre_proc[idx_ope + num_ope_bias][idx_ope + num_ope_bias + 1] = True
            if idx_ope != 0:
                vector = torch.zeros(matrix_cal_cumul.size(0))
                vector[idx_ope + num_ope_bias - 1] = 1
                matrix_cal_cumul[:, idx_ope + num_ope_bias] = matrix_cal_cumul[:, idx_ope + num_ope_bias - 1] + vector
            flag += 1
        # not proc_time (machine)
        elif flag_time == 0:
            mac = x - 1
            flag += 1
            flag_time = 1
        # proc_time
        else:
            matrix_proc_time[idx_ope + num_ope_bias][mac] = x
            flag += 1
            flag_time = 0
    return num_ope
