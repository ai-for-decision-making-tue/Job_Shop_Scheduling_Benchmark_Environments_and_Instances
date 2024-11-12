import argparse
import logging
import os

from plotting.drawer import plot_gantt_chart
from scheduling_environment.jobShop import JobShop
from solution_methods.dispatching_rules.utils import configure_simulation_env, output_dir_exp_name, results_saving
from solution_methods.helper_functions import load_parameters, load_job_shop_env
from solution_methods.dispatching_rules.src.scheduling_functions import scheduler

logging.basicConfig(level=logging.INFO)
PARAM_FILE = "../../configs/dispatching_rules.toml"


def run_dispatching_rules(jobShopEnv, **kwargs):
    dispatching_rule = kwargs['instance']['dispatching_rule']
    machine_assignment_rule = kwargs['instance']['machine_assignment_rule']

    if dispatching_rule == 'SPT' and machine_assignment_rule != 'SPT':
        raise ValueError("SPT dispatching rule requires SPT machine assignment rule.")

    # Configure simulation environment
    simulationEnv = configure_simulation_env(jobShopEnv, **kwargs)
    simulationEnv.simulator.process(scheduler(simulationEnv, **kwargs))

    # For online arrivals, run the simulation until the configured end time
    if kwargs['instance']['online_arrivals']:
        simulationEnv.simulator.run(until=kwargs['online_arrival_details']['simulation_time'])
    # For static instances, run until all operations are scheduled
    else:
        simulationEnv.simulator.run()

    makespan = simulationEnv.jobShopEnv.makespan
    logging.info(f"Makespan: {makespan}")

    return makespan, simulationEnv.jobShopEnv


def main(param_file: str = PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    # Configure the simulation environment
    if parameters['instance']['online_arrivals']:
        jobShopEnv = JobShop()
        makespan, jobShopEnv = run_dispatching_rules(jobShopEnv, **parameters)
        logging.warning(f"Makespan objective is irrelevant for problems configured with 'online arrivals'.")
    else:
        jobShopEnv = load_job_shop_env(parameters['instance'].get('problem_instance'))
        makespan, jobShopEnv = run_dispatching_rules(jobShopEnv, **parameters)

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
    parser = argparse.ArgumentParser(description="Run Dispatching Rules.")
    parser.add_argument(
        "-f",
        "--config_file",
        type=str,
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)