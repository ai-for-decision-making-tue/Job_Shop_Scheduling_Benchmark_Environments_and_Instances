import argparse
import logging
import multiprocessing

from deap import tools

from plotting.drawer import draw_gantt_chart, draw_precedence_relations
from solution_methods.genetic_algorithm.operators import (
    evaluate_individual, evaluate_population, repair_precedence_constraints, variation)
from solution_methods.genetic_algorithm.helper_functions import record_stats
from solution_methods.helper_functions import load_parameters
from solution_methods.genetic_algorithm.run_initialization import initialize_run

logging.basicConfig(level=logging.INFO)

PARAM_FILE = "configs/genetic_algorithm.toml"
DEFAULT_RESULTS_ROOT = "./results/single_runs"


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

    exp_name = ("/seed" + str(parameters['algorithm']["seed"]) + "/")
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
