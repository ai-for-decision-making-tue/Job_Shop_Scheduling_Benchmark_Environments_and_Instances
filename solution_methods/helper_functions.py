import os
import random
import numpy as np

import pandas as pd
import tomli
import torch

from data_parsers import parser_fajsp, parser_fjsp, parser_fjsp_sdst, parser_jsp_fsp
from scheduling_environment.jobShop import JobShop


def load_parameters(config_toml):
    """Load parameters from a toml file"""
    with open(config_toml, "rb") as f:
        config_params = tomli.load(f)
    return config_params


def load_job_shop_env(problem_instance: str, from_absolute_path=False) -> JobShop:
    jobShopEnv = JobShop()
    if '/fsp/' in problem_instance or '/jsp/' in problem_instance:
        jobShopEnv = parser_jsp_fsp.parse(jobShopEnv, problem_instance, from_absolute_path)
    elif '/fjsp/' in problem_instance:
        jobShopEnv = parser_fjsp.parse(jobShopEnv, problem_instance, from_absolute_path)
    elif '/fjsp_sdst/' in problem_instance:
        jobShopEnv = parser_fjsp_sdst.parse(jobShopEnv, problem_instance, from_absolute_path)
    elif '/fajsp/' in problem_instance:
        jobShopEnv = parser_fajsp.parse(jobShopEnv, problem_instance, from_absolute_path)
    else:
        raise NotImplementedError(
            f"""Problem instance {
            problem_instance
            } not implemented"""
        )
    jobShopEnv._name = problem_instance
    return jobShopEnv


def set_seeds(seed_value=0):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def initialize_device(parameters: dict) -> torch.device:
    device_str = "cpu"
    if parameters['test_parameters']['device'] == "cuda":
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def update_operations_available_for_scheduling(env):
    scheduled_operations = set(env.scheduled_operations)
    precedence_relations = env.precedence_relations_operations
    operations_available = [
        operation
        for operation in env.operations
        if operation not in scheduled_operations and all(
            prec_operation in scheduled_operations
            for prec_operation in precedence_relations[operation.operation_id]
        )
    ]
    env.set_operations_available_for_scheduling(operations_available)
