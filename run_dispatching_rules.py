import argparse
import logging

from data_parsers import parser_fajsp, parser_fjsp, parser_jsp_fsp
from plotting.drawer import draw_gantt_chart, draw_precedence_relations
from scheduling_environment.simulationEnv import SimulationEnv
from solution_methods.helper_functions import load_parameters
from solution_methods.dispatching_rules.scheduler import schedule_operations

PARAM_FILE = "configs/dispatching_rules.toml"


def run_method(simulationEnv, dispatching_rule, machine_assignment_rule):
    """Schedule simulator and schedule operations with the dispatching rules"""

    if simulationEnv.online_arrivals:
        # Start the online job generation process
        simulationEnv.simulator.process(simulationEnv.generate_online_job_arrivals())

        # Run the scheduling_environment until all operations are processed
        while True:
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule)
            yield simulationEnv.simulator.timeout(1)

    else:
        # add machine resources to the environment
        for _ in simulationEnv.JobShop.machines:
            simulationEnv.add_machine_resources()

        # Run the scheduling env and schedule operations until all operations are processed from the data instance
        while len(simulationEnv.processed_operations) < sum([len(job.operations) for job in simulationEnv.JobShop.jobs]):
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule)
            yield simulationEnv.simulator.timeout(1)


def main(param_file: str = PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    simulationEnv = SimulationEnv(online_arrivals=parameters['instance']['online_arrivals'])

    if not parameters['instance']['online_arrivals']:
        try:
            if 'fjsp' in parameters['instance']['problem_instance']:
                simulationEnv.JobShop = parser_fjsp.parse(simulationEnv.JobShop,
                                                          parameters['instance']['problem_instance'])
            elif 'fajsp' in parameters['instance']['problem_instance']:
                simulationEnv.JobShop = parser_fajsp.parse(simulationEnv.JobShop,
                                                           parameters['instance']['problem_instance'])
            elif 'jsp' in parameters['instance']['problem_instance']:
                simulationEnv.JobShop = parser_jsp_fsp.parse(simulationEnv.JobShop,
                                                             parameters['instance']['problem_instance'])
        except Exception as e:
            print(f"Only able to schedule '/fjsp/, '/fasjp/ or '/jsp/' jobs': {e}")

        simulationEnv.simulator.process(
            run_method(simulationEnv, parameters['instance']['dispatching_rule'],
                           parameters['instance']['machine_assignment_rule']))
        simulationEnv.simulator.run()
        logging.info(f"Makespan: {simulationEnv.JobShop.makespan}")

    else:
        simulationEnv.set_online_arrival_details(parameters['online_arrival_details'])
        simulationEnv.JobShop.set_nr_of_machines(parameters['online_arrival_details']['number_total_machines'])
        simulationEnv.simulator.process(run_method(simulationEnv, parameters['instance']['dispatching_rule'],
                                        parameters['instance']['machine_assignment_rule']))
        simulationEnv.simulator.run(until=parameters['online_arrival_details']['simulation_time'])

    if parameters['output']['plotting']:
        draw_precedence_relations(simulationEnv.JobShop)
        draw_gantt_chart(simulationEnv.JobShop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run (Online) Job Shop Simulation Model")
    parser.add_argument(
        "-f",
        "--config_file",
        type=str,
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)