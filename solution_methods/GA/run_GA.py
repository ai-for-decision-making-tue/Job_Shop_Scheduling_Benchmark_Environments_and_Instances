import argparse
import logging
import os

from deap import tools

from solution_methods.helper_functions import load_parameters, load_job_shop_env
from solution_methods.GA.src.operators import (evaluate_individual, evaluate_population, repair_precedence_constraints, variation)
from solution_methods.GA.utils import record_stats, output_dir_exp_name, results_saving
from solution_methods.GA.src.initialization import initialize_run

from plotting.drawer import plot_gantt_chart

logging.basicConfig(level=logging.INFO)
PARAM_FILE = "../../configs/GA.toml"


def run_GA(jobShopEnv, population, toolbox, stats, hof, **kwargs):
    """Executes the genetic algorithm and returns the best individual.

    Args:
        jobShopEnv: The problem environment.
        population: The initial population.
        toolbox: DEAP toolbox.
        stats: DEAP statistics.
        hof: Hall of Fame.
        kwargs: Additional keyword arguments.

    Returns:
        The best individual found by the genetic algorithm.
    """

    # Initial population setup for Hall of Fame and statistics
    hof.update(population)

    gen = 0
    logbook = tools.Logbook()
    logbook.header = ["gen"] + (stats.fields if stats else [])
    df_list = []

    # Initial statistics recording
    record_stats(gen, population, logbook, stats, kwargs['output']['logbook'], df_list, logging)
    if kwargs['output']['logbook']:
        logging.info(logbook.stream)

    for gen in range(1, kwargs['algorithm']['ngen'] + 1):
        # Vary the population
        offspring = variation(population, toolbox,
                              pop_size=kwargs['algorithm'].get('population_size'),
                              cr = kwargs['algorithm'].get('cr'),
                              indpb = kwargs['algorithm'].get('indpb'))

        # Repair precedence constraints if the environment requires it (only for assembly scheduling (fajsp))
        if any(keyword in jobShopEnv.instance_name for keyword in ['/dafjs/', '/yfjs/']):
            try:
                offspring = repair_precedence_constraints(jobShopEnv, offspring)
            except Exception as e:
                logging.error(f"Error repairing precedence constraints: {e}")
                continue

        # Evaluate offspring fitness
        try:
            fitnesses = evaluate_population(toolbox, offspring)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit
        except Exception as e:
            logging.error(f"Error evaluating offspring fitness: {e}")
            continue

        # Select the next generation
        population[:] = toolbox.select(population + offspring)

        # Update Hall of Fame and statistics with the new generation
        hof.update(population)
        record_stats(gen, population, logbook, stats, kwargs['output']['logbook'], df_list, logging)

    makespan, jobShopEnv = evaluate_individual(hof[0], jobShopEnv, reset=False)
    logging.info(f"Makespan: {makespan}")
    return makespan, jobShopEnv


def main(param_file=PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
        logging.info(f"Parameters loaded from {param_file}.")
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    # Load the job shop environment, and initialize the genetic algorithm
    jobShopEnv = load_job_shop_env(parameters['instance'].get('problem_instance'))
    population, toolbox, stats, hof = initialize_run(jobShopEnv, **parameters)
    makespan, jobShopEnv = run_GA(jobShopEnv, population, toolbox, stats, hof, **parameters)

    if makespan is not None:
        # Check output configuration and prepare output paths if needed
        output_config = parameters['output']
        save_gantt = output_config.get('save_gantt')
        save_results = output_config.get('save_results')
        show_gantt = output_config.get('show_gantt')

        if save_gantt or save_results:
            output_dir, exp_name = output_dir_exp_name(parameters)
            output_dir = os.path.join(output_dir, f"{exp_name}")
            os.makedirs(output_dir, exist_ok=True)

        # Plot Gantt chart if required
        if show_gantt or save_gantt:
            logging.info("Generating Gantt chart.")
            plt = plot_gantt_chart(jobShopEnv)

            if save_gantt:
                plt.savefig(output_dir + "/gantt.png")
                logging.info(f"Gantt chart saved to {output_dir}")

            if show_gantt:
                plt.show()

        # Save results if enabled
        if save_results:
            results_saving(makespan, output_dir, parameters)
            logging.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA")
    parser.add_argument(
        "-f", "--config_file",
        type=str,
        default=PARAM_FILE,
        help="path to configuration file",
    )
    args = parser.parse_args()
    main(param_file=args.config_file)
