import argparse
import logging

from scheduling_environment.simulationEnv import SimulationEnv
from data_parsers import parser_fajsp, parser_fjsp
from solutions.dispatching_rules.helper_functions import *
from solutions.helper_functions import load_parameters
from plotting.drawer import draw_gantt_chart, draw_precedence_relations

PARAM_FILE = "configs/dispatching_rules.toml"


def select_operation(simulationEnv, machine, dispatching_rule, machine_assignment_rule):
    """use dispatching rules to select the next operation to schedule"""
    operation_priorities = {}

    # Calculate the priority values for the operations based on the dispatching rule and machine assignment rule
    for job in simulationEnv.JobShop.jobs:
        for operation in job.operations:
            if operation not in simulationEnv.processed_operations and operation not in simulationEnv.JobShop.scheduled_operations and machine.machine_id in operation.processing_times:
                if check_precedence_relations(simulationEnv, operation):
                    if dispatching_rule == 'FIFO' and machine_assignment_rule == 'SPT':
                        min_processing_time = min(operation.processing_times.values())
                        min_keys = [key for key, value in operation.processing_times.items() if
                                    value == min_processing_time]
                        if machine.machine_id in min_keys:
                            operation_priorities[operation] = operation.job_id

                    elif dispatching_rule in ['MOR', 'LOR'] and machine_assignment_rule == 'SPT':
                        min_processing_time = min(operation.processing_times.values())
                        min_keys = [key for key, value in operation.processing_times.items() if
                                    value == min_processing_time]
                        if machine.machine_id in min_keys:
                            operation_priorities[operation] = get_operations_remaining(simulationEnv, operation)

                    elif dispatching_rule in ['MWR', 'LWR'] and machine_assignment_rule == 'SPT':
                        min_processing_time = min(operation.processing_times.values())
                        min_keys = [key for key, value in operation.processing_times.items() if
                                    value == min_processing_time]
                        if machine.machine_id in min_keys:
                            operation_priorities[operation] = get_work_remaining(simulationEnv, operation)

                    elif dispatching_rule == 'FIFO' and machine_assignment_rule == 'EET':
                        earliest_end_time_machines = simulationEnv.get_earliest_end_time_machines(operation)
                        if machine.machine_id in earliest_end_time_machines:
                            operation_priorities[operation] = operation.job_id

                    elif dispatching_rule in ['MOR', 'LOR'] and machine_assignment_rule == 'EET':
                        earliest_end_time_machines = get_earliest_end_time_machines(simulationEnv, operation)
                        if machine.machine_id in earliest_end_time_machines:
                            operation_priorities[operation] = get_operations_remaining(simulationEnv, operation)

                    elif dispatching_rule in ['MWR', 'LWR'] and machine_assignment_rule == 'EET':
                        earliest_end_time_machines = simulationEnv.get_earliest_end_time_machines(operation)
                        if machine.machine_id in earliest_end_time_machines:
                            operation_priorities[operation] = simulationEnv.get_work_remaining(operation)

    if len(operation_priorities) == 0:
        return None
    else:
        if dispatching_rule == 'FIFO' or dispatching_rule == 'LOR' or dispatching_rule == 'LWR':
            return min(operation_priorities, key=operation_priorities.get)
        else:
            return max(operation_priorities, key=operation_priorities.get)


def schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule):
    """Schedule operations on the machines based on the priority values"""
    machines_available = [machine for machine in simulationEnv.JobShop.machines if
                          simulationEnv.machine_resources[machine.machine_id].count == 0]
    machines_available.sort(key=lambda m: m.machine_id)

    for machine in machines_available:
        operation_to_schedule = select_operation(simulationEnv, machine, dispatching_rule, machine_assignment_rule)
        if operation_to_schedule is not None:
            simulationEnv.JobShop._scheduled_operations.append(operation_to_schedule)
            # Check if all precedence relations are satisfied
            simulationEnv.simulator.process(simulationEnv.perform_operation(operation_to_schedule, machine))


def run_simulation(simulationEnv, dispatching_rule, machine_assignment_rule):
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

        # Run the scheduling_environment and schedule operations until all operations are processed from the data instance
        while len(simulationEnv.processed_operations) < sum([len(job.operations) for job in simulationEnv.JobShop.jobs]):
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule)
            yield simulationEnv.simulator.timeout(1)


def main(param_file: str = PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    simulationEnv = SimulationEnv(
        online_arrivals=parameters['instance']['online_arrivals'],
    )

    if not parameters['instance']['online_arrivals']:
        try:
            if 'fjsp' in parameters['instance']['problem_instance']:
                simulationEnv.JobShop = parser_fjsp.parse(simulationEnv.JobShop, parameters['instance']['problem_instance'])
            elif 'fajsp' in parameters['instance']['problem_instance']:
                simulationEnv.JobShop = parser_fajsp.parse(simulationEnv.JobShop, parameters['instance']['problem_instance'])
        except Exception as e:
            print(f"Only able to scheduled '/fjsp/ or 'fasjp/ jobs': {e}")
        simulationEnv.simulator.process(run_simulation(simulationEnv, parameters['instance']['dispatching_rule'], parameters['instance']['machine_assignment_rule']))
        simulationEnv.simulator.run()
        logging.info(f"Makespan: {simulationEnv.JobShop.makespan}")

    else:
        simulationEnv.set_online_arrival_details(parameters['online_arrival_details'])
        simulationEnv.JobShop.set_nr_of_machines(parameters['online_arrival_details']['number_total_machines'])
        simulationEnv.simulator.process(run_simulation(simulationEnv, parameters['instance']['dispatching_rule'], parameters['instance']['machine_assignment_rule']))
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
        help="path to config JSON",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)
