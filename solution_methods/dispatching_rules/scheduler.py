from solution_methods.dispatching_rules.rules import *
from solution_methods.dispatching_rules.helper_functions import *


def select_operation(simulationEnv, machine, dispatching_rule, machine_assignment_rule):
    """ Use dispatching rules to select the next operation to schedule """
    operation_priorities = {}

    for job in simulationEnv.JobShop.jobs:
        for operation in job.operations:
            if operation not in simulationEnv.processed_operations and operation not in simulationEnv.JobShop.scheduled_operations and machine.machine_id in operation.processing_times:
                if check_precedence_relations(simulationEnv, operation):
                    if machine_assignment_rule == 'SPT' and spt_rule(operation, machine.machine_id):
                        if dispatching_rule == 'FIFO':
                            operation_priorities[operation] = fifo_priority(operation)
                        elif dispatching_rule == 'MOR':
                            operation_priorities[operation] = mor_priority(simulationEnv, operation)
                        elif dispatching_rule == 'MWR':
                            operation_priorities[operation] = mwr_priority(simulationEnv, operation)
                        elif dispatching_rule == 'LOR':
                            operation_priorities[operation] = lor_priority(simulationEnv, operation)
                        elif dispatching_rule == 'LWR':
                            operation_priorities[operation] = lwr_priority(simulationEnv, operation)
                    elif machine_assignment_rule == 'EET' and eet_rule(simulationEnv, operation, machine.machine_id):
                        if dispatching_rule == 'FIFO':
                            operation_priorities[operation] = fifo_priority(operation)
                        elif dispatching_rule == 'MOR':
                            operation_priorities[operation] = mor_priority(simulationEnv, operation)
                        elif dispatching_rule == 'MWR':
                            operation_priorities[operation] = mwr_priority(simulationEnv, operation)
                        elif dispatching_rule == 'LOR':
                            operation_priorities[operation] = lor_priority(simulationEnv, operation)
                        elif dispatching_rule == 'LWR':
                            operation_priorities[operation] = lwr_priority(simulationEnv, operation)

    if not operation_priorities:
        return None
    else:
        if dispatching_rule in ['FIFO', 'LOR', 'LWR']:
            return min(operation_priorities, key=operation_priorities.get)
        else:
            return max(operation_priorities, key=operation_priorities.get)


def schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule):
    """Schedule operations on the machines based on the priority values"""
    machines_available = [machine for machine in simulationEnv.JobShop.machines if
                          simulationEnv.machine_resources[machine.machine_id].count == 0]
    machines_available.sort(key=lambda m: m.machine_id)

    for machine in machines_available:
        operation_to_schedule = select_operation(simulationEnv, machine, dispatching_rule,
                                                 machine_assignment_rule)
        if operation_to_schedule is not None:
            simulationEnv.JobShop._scheduled_operations.append(operation_to_schedule)
            simulationEnv.simulator.process(simulationEnv.perform_operation(operation_to_schedule, machine))