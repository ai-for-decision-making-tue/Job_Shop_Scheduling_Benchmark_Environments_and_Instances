from solution_methods.dispatching_rules.src.helper_functions import *


def fifo_priority(operation):
    """ FIFO Rule: First In, First Out """
    return operation.job_id


def spt_priority(operation):
    """ SPT Rule: Shortest Processing Time """
    return min(operation.processing_times.values())


def mor_priority(simulationEnv, operation):
    """ MOR Rule: Most Operations Remaining """
    return get_operations_remaining(simulationEnv, operation)


def lor_priority(simulationEnv, operation):
    """ LOR Rule: Least Operations Remaining """
    return get_operations_remaining(simulationEnv, operation)


def mwr_priority(simulationEnv, operation):
    """ MWR Rule: Most Work Remaining """
    return get_work_remaining(simulationEnv, operation)


def lwr_priority(simulationEnv, operation):
    """ LWR Rule: Least Work Remaining """
    return get_work_remaining(simulationEnv, operation)


def spt_rule(operation, machine_id):
    """ SPT Rule: Shortest Processing Time """
    min_processing_time = min(operation.processing_times.values())
    min_keys = [key for key, value in operation.processing_times.items() if value == min_processing_time]
    return machine_id in min_keys


def eet_rule(simulationEnv, operation, machine_id):
    """ EET Rule: Earliest End Time """
    earliest_end_time_machines = get_earliest_end_time_machines(simulationEnv, operation)
    return machine_id in earliest_end_time_machines