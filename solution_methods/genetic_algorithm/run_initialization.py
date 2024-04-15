import logging
from multiprocessing.pool import Pool

import numpy as np
from deap import base, creator, tools

from solution_methods.genetic_algorithm.operators import (
    evaluate_individual, evaluate_population, init_individual, init_population, mutate_sequence_exchange,
    mutate_shortest_proc_time, pox_crossover)
from solution_methods.helper_functions import load_job_shop_env, set_seeds


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

    set_seeds(kwargs["algorithm"]["seed"])

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