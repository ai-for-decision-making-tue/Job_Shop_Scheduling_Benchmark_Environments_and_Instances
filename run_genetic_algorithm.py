import argparse
import logging
import multiprocessing
from multiprocessing.pool import Pool

import numpy as np
from deap import base, creator, tools

from plotting.drawer import draw_gantt_chart, draw_precedence_relations
from solution_methods.genetic_algorithm.operators import (
    evaluate_individual, evaluate_population, init_individual, init_population, mutate_sequence_exchange,
    mutate_shortest_proc_time, pox_crossover, repair_precedence_constraints, variation)
from solution_methods.helper_functions import dict_to_excel, load_job_shop_env, load_parameters, record_stats

logging.basicConfig(level=logging.INFO)

PARAM_FILE = "configs/genetic_algorithm.toml"
DEFAULT_RESULTS_ROOT = "./results/single_runs"


def initialize_run(pool: Pool, **kwargs):
    """Initializes the run by setting up the environment, toolbox, statistics, hall of fame, and initial population.

    Args:
        pool: Multiprocessing pool.
        kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the initial population, toolbox, statistics, hall of fame, and environment.
    """
    try:
        jobShopEnv = load_job_shop_env(kwargs['instance']['problem_instance'])
    except FileNotFoundError:
        logging.error(f"Problem instance {kwargs['instance']['problem_instance']} not found.")
        return

    toolbox = base.Toolbox()
    toolbox.register("map", pool.map)
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox.register("init_individual", init_individual, creator.Individual, kwargs, jobShopEnv=jobShopEnv)
    toolbox.register("mate_TwoPoint", tools.cxTwoPoint)
    toolbox.register("mate_Uniform", tools.cxUniform, indpb=0.5)
    toolbox.register("mate_POX", pox_crossover, nr_preserving_jobs=1)

    toolbox.register("mutate_machine_selection", mutate_shortest_proc_time, jobShopEnv=jobShopEnv)
    toolbox.register("mutate_operation_sequence", mutate_sequence_exchange)
    toolbox.register("select", tools.selTournament, k=kwargs['algorithm']['population_size'], tournsize=3)
    toolbox.register("evaluate_individual", evaluate_individual, jobShopEnv=jobShopEnv)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    hof = tools.HallOfFame(1)

    initial_population = init_population(toolbox, kwargs['algorithm']['population_size'], )
    try:
        fitnesses = evaluate_population(toolbox, initial_population, logging)
    except Exception as e:
        logging.error(f"An error occurred during initial population evaluation: {e}")
        return

    for ind, fit in zip(initial_population, fitnesses):
        ind.fitness.values = fit

    return initial_population, toolbox, stats, hof, jobShopEnv


def run_method(jobShopEnv, population, toolbox, folder, exp_name, stats=None, hof=None, **kwargs):
    """Executes the genetic algorithm and returns the best individual.

    Args:
        jobShopEnv: The problem environment.
        population: The initial population.
        toolbox: DEAP toolbox.
        folder: The folder to save results in.
        exp_name: The experiment name.
        stats: DEAP statistics (optional).
        hof: Hall of Fame (optional).
        kwargs: Additional keyword arguments.

    Returns:
        The best individual found by the genetic algorithm.
    """

    try:
        if kwargs['output']['plotting']:
            draw_precedence_relations(jobShopEnv)

        hof.update(population)

        gen = 0
        df_list = []
        logbook = tools.Logbook()
        logbook.header = ["gen"] + (stats.fields if stats else [])

        # Update the statistics with the new population
        record_stats(gen, population, logbook, stats, kwargs['output']['logbook'], df_list, logging)

        if kwargs['output']['logbook']:
            logging.info(logbook.stream)

        for gen in range(1, kwargs['algorithm']['ngen'] + 1):
            # Vary the population
            offspring = variation(population, toolbox, kwargs['algorithm']['population_size'], kwargs['algorithm']['cr'], kwargs['algorithm']['indpb'])

            # Ensure that precedence constraints between jobs are satisfied (only for assembly scheduling (fajsp))
            if '/dafjs/' or '/yfjs/' in jobShopEnv.instance_name:
                offspring = repair_precedence_constraints(jobShopEnv, offspring)

            # Evaluate the population
            fitnesses = evaluate_population(toolbox, offspring, logging)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            hof.update(offspring)

            # Select next generation population
            population[:] = toolbox.select(population + offspring)
            # Update the statistics with the new population
            record_stats(gen, population, logbook, stats, kwargs['output']['logbook'], df_list, logging)
            make_span, jobShopEnv = evaluate_individual(hof[0], jobShopEnv, reset=False)

        if kwargs['output']['plotting']:
            make_span, jobShopEnv = evaluate_individual(hof[0], jobShopEnv, reset=False)
            logging.info('make_span: %s', jobShopEnv.makespan)
            draw_gantt_chart(jobShopEnv)

        return hof[0]

    except Exception as e:
        logging.error(f"An error occurred during the algorithm run: {e}")


def main(param_file=PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return
    
    pool = multiprocessing.Pool()
    folder = (
            DEFAULT_RESULTS_ROOT
            + "/"
            + str(parameters['instance']['problem_instance'])
            + "/ngen"
            + str(parameters['algorithm']["ngen"])
            + "_pop"
            + str(parameters['algorithm']['population_size'])
            + "_cr"
            + str(parameters['algorithm']["cr"])
            + "_indpb"
            + str(parameters['algorithm']["indpb"])
    )

    exp_name = ("/rseed" + str(parameters['algorithm']["rseed"]) + "/")
    population, toolbox, stats, hof, jobShopEnv = initialize_run(pool, **parameters)
    best_individual = run_method(jobShopEnv, population, toolbox, folder, exp_name, stats, hof, **parameters)
    return best_individual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA")
    parser.add_argument(
        "config_file",
        metavar='-f',
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)
