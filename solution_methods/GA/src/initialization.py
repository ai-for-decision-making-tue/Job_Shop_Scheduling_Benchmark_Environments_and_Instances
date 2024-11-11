import logging

import multiprocessing

# from multiprocessing.pool import Pool

import numpy as np
from deap import base, creator, tools

from solution_methods.GA.src.operators import (
    evaluate_individual, evaluate_population, init_individual, init_population,
    mutate_sequence_exchange, mutate_shortest_proc_time, pox_crossover)
from solution_methods.helper_functions import set_seeds


def initialize_run(jobShopEnv, **kwargs):
    """
    Initializes the GA run by setting up the DEAP toolbox, statistics, hall of fame, and initial population.

    Args:
        jobShopEnv: The job shop environment to be optimized.
        pool: Multiprocessing pool for parallel processing.
        kwargs: Additional keyword arguments for setting algorithm parameters.

    Returns:
        tuple: (initial_population, toolbox, stats, hof)
            - initial_population: Initialized population.
            - toolbox: DEAP toolbox with registered operators.
            - stats: Statistics object for tracking evolution progress.
            - hof: Hall of fame to store the best individuals.
    """
    # Set random seed
    set_seeds(kwargs["algorithm"].get("seed", None))

    # Initialize logging if not already configured
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO)

    # Set up DEAP creator classes
    if not hasattr(creator, "Fitness"):
        creator.create("Fitness", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.Fitness)

    # Define and register operators and functions in the DEAP toolbox
    toolbox = base.Toolbox()

    # Initialize the multiprocessing pool
    if kwargs['algorithm']['multiprocessing']:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

    # Register individual and genetic operators
    toolbox.register("init_individual", init_individual, creator.Individual, jobShopEnv=jobShopEnv)
    toolbox.register("mate_TwoPoint", tools.cxTwoPoint)
    toolbox.register("mate_Uniform", tools.cxUniform, indpb=0.5)
    toolbox.register("mate_POX", pox_crossover, nr_preserving_jobs=1)
    toolbox.register("mutate_machine_selection", mutate_shortest_proc_time, jobShopEnv=jobShopEnv)
    toolbox.register("mutate_operation_sequence", mutate_sequence_exchange)
    toolbox.register("select", tools.selTournament, k=kwargs['algorithm']['population_size'], tournsize=3)
    toolbox.register("evaluate_individual", evaluate_individual, jobShopEnv=jobShopEnv)

    # Setup statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Create Hall of Fame to track the best individuals
    hof = tools.HallOfFame(1)

    # try:
    initial_population = init_population(toolbox, kwargs['algorithm']['population_size'], )
    fitnesses = evaluate_population(toolbox, initial_population)

    # Assign fitness values to individuals
    for ind, fit in zip(initial_population, fitnesses):
        ind.fitness.values = fit

    # except Exception as e:
    #     logging.error(f"An error occurred during initial population evaluation: {e}")
    #     return None, None, None, None

    return initial_population, toolbox, stats, hof
