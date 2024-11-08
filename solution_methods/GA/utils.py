import os
import datetime
import json

import pandas as pd

DEFAULT_RESULTS_ROOT = os.getcwd() + "/results/GA/"


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


def output_dir_exp_name(parameters):
    if 'experiment_name' in parameters['output'] is not None:
        exp_name = parameters['output']['experiment_name']
    else:
        instance_name = parameters['instance']['problem_instance'].replace('/', '_')[1:]
        instance_name = instance_name.split('.')[0] if '.' in instance_name else instance_name
        population_size = parameters['algorithm'].get('population_size')
        ngen = parameters['algorithm'].get('ngen')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{instance_name}_pop_{population_size}_ngen_{ngen}_{timestamp}"

    if 'folder_name' in parameters['output'] is not None:
        output_dir = parameters['output']['folder_name']
    else:
        output_dir = DEFAULT_RESULTS_ROOT
    return output_dir, exp_name


def results_saving(makespan, path, parameters):
    """
    Save the GA optimization results to a JSON file.
    """
    results = {
        "instance": parameters["instance"]["problem_instance"],
        "makespan": makespan,
        "ngen": parameters["algorithm"]["ngen"],
        "population_size": parameters["algorithm"]["population_size"],
        "crossover_rate": parameters["algorithm"]["cr"],
        "mutation_rate": parameters["algorithm"]["indpb"]
    }

    # Generate a default experiment name based on instance and solve time if not provided
    os.makedirs(path, exist_ok=True)

    # Save results to JSON
    file_path = os.path.join(path, "GA_results.json")
    with open(file_path, "w") as outfile:
        json.dump(results, outfile, indent=4)
