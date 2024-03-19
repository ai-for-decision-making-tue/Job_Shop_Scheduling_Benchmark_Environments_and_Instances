import os
import tomli

import pandas as pd

from scheduling_environment.jobShop import JobShop
from data_parsers import parser_fajsp, parser_fjsp, parser_jsp_fsp, parser_fjsp_sdst


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


def create_stats_list(population, gen):
    stats_list = []
    for ind in population:
        tmp_dict = {}
        tmp_dict.update(
            {
                "Generation": gen,
                "obj1": ind.fitness.values[0]
            })
        if hasattr(ind, "objectives"):
            tmp_dict.update(
                {
                    "obj1": ind.objectives[0],
                }
            )
        tmp_dict = {**tmp_dict}
        stats_list.append(tmp_dict)
    return stats_list


def record_stats(gen, population, logbook, stats, verbose, df_list, logging):
    stats_list = create_stats_list(population, gen)
    df_list.append(pd.DataFrame(stats_list))
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=gen, **record)
    if verbose:
        logging.info(logbook.stream)


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


def dict_to_excel(dictionary, folder, filename):
    """Save outputs in files"""

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Check if the file exists, if so, give a warning
    full_path = os.path.join(folder, filename)
    if os.path.exists(full_path):
        print(f"Warning: {full_path} already exists. Overwriting file.")

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([dictionary])

    # Save the DataFrame to Excel
    df.to_excel(full_path, index=False, engine='openpyxl')
