import random
import sys
import os
import time
import pandas as pd
import numpy as np
sys.path.append(".")
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.plotting.drawer import plot_gantt_chart, draw_precedence_relations
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.data_parsers.parser_fjsp import parse_fjsp
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.data_parsers.parser_jsp_fsp import parse_jsp_fsp
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.scheduling_environment.jobShop import JobShop

from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.helper_functions import load_job_shop_env, load_parameters
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.dispatching_rules.run_dispatching_rules import run_dispatching_rules
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.GA.src.initialization import initialize_run
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.GA.run_GA import run_GA
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.MILP.run_MILP import run_MILP
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.FJSP_DRL.run_FJSP_DRL import run_FJSP_DRL

from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.scheduling_environment.jobShop import JobShop
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.scheduling_environment.myJobShop import MyJobShop
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.plotting.drawer import plot_gantt_chart, draw_precedence_relations
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.dispatching_rules.utils import configure_simulation_env, output_dir_exp_name, results_saving
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.helper_functions import load_parameters, load_job_shop_env
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.dispatching_rules.src.scheduling_functions import (
    scheduler, schedule_operations, select_operation, scheduler_step, scheduler_random_dispatching_rule,
    select_operation_and_machine_fast)
import logging
from multiprocessing import Pool
import itertools


def fast_scheduling(jobShopEnv,**kwargs):
    dispatching_rule = kwargs['dispatching_rule']
    machine_assignment_rule = kwargs['machine_assignment_rule']
    # dispatching rule candidates for random dispatching rule
    #set seed for random dispatching rule
    np.random.seed(kwargs['seed'])
    dispatching_rule_candidates = [('FIFO', 'EET'), ('SPT', 'SPT'), ('MOR', 'EET'), ('MWR', 'EET'), ('LOR', 'EET'), ('LWR', 'EET')] 

    print(dispatching_rule, machine_assignment_rule)
    if dispatching_rule == 'SPT' and machine_assignment_rule != 'SPT':
        raise ValueError("SPT dispatching rule requires SPT machine assignment rule.")

    for i, operation in enumerate(jobShopEnv.operations):
        jobShopEnv._scheduled_operation_indics[operation.operation_id] = 0
    
    jobShopEnv.update_operations_available_for_scheduling_fast()
    while len(jobShopEnv.scheduled_operations) < jobShopEnv.nr_of_operations:
        # 1. while there are available machines, select operation and machine
        # 2. process operation on machine
        # 3. add time, check if operation is available, finished or not

        while len(jobShopEnv.get_available_machines(jobShopEnv._now)) > 0:
            if dispatching_rule == 'random_dispatching_rule':
                action_dispatching_rule, action_machine_assignment_rule = random.choice(dispatching_rule_candidates)
            else:
                action_dispatching_rule = dispatching_rule
                action_machine_assignment_rule = machine_assignment_rule

            operation, machine_id, duration = select_operation_and_machine_fast(jobShopEnv,dispatching_rule=action_dispatching_rule, machine_assignment_rule=action_machine_assignment_rule)
            if operation is None:
                break
            jobShopEnv.add_operation_to_machine(operation, machine_id, duration)
            jobShopEnv.update_operations_available_for_scheduling_fast()

        while len(jobShopEnv.operations_available_for_scheduling) <=0:
            machine_state = jobShopEnv.schedule_operation_on_machine_step_fast()
            jobShopEnv.update_operations_available_for_scheduling_fast()
            if any(state is not None for state in machine_state):
                break

    makepsan = max(operation.scheduling_information['end_time'] for operation in jobShopEnv.scheduled_operations)
    return makepsan, jobShopEnv

problem_instance_path = ""
kwargs = {}
jobShopEnv = load_job_shop_env(MyJobShop, from_absolute_path=True, problem_instance=problem_instance_path)
makespan, jobShopEnv = fast_scheduling(jobShopEnv, **kwargs)