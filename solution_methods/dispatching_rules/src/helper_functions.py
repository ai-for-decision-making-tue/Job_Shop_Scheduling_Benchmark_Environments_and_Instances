def get_operations_remaining(simulationEnv, operation):
    """get remaining operations of the job"""
    return len(
        [operation for operation in operation.job.operations if operation not in simulationEnv.processed_operations])


def get_work_remaining(simulationEnv, operation):
    """get amount of work remaining of the job (average taken of different machine options)"""
    operations_remaining = [operation for operation in operation.job.operations if
                            operation not in simulationEnv.processed_operations]
    return sum([sum(operation.processing_times.values()) / len(operation.processing_times) for operation in
                operations_remaining])


def get_earliest_end_time_machines(simulationEnv, operation):
    """get earliest end time of machines, when operation would be scheduled on it"""
    finish_times = {}
    machine_options = operation.processing_times.keys()
    for machine_option in machine_options:
        machine = simulationEnv.jobShopEnv.get_machine(machine_option)
        if machine.scheduled_operations == []:
            finish_times[machine_option] = simulationEnv.simulator.now + operation.processing_times[machine_option]
        else:
            if simulationEnv.jobShopEnv._sequence_dependent_setup_times != []:
                finish_times[machine_option] = operation.processing_times[machine_option] \
                                               + machine._processed_operations[-1].scheduling_information['end_time'] \
                                               + simulationEnv.jobShopEnv._sequence_dependent_setup_times[machine.machine_id][
                                                machine.scheduled_operations[-1].operation_id][operation.operation_id]
            else:
                finish_times[machine_option] = operation.processing_times[machine_option] \
                                               + machine._processed_operations[-1].scheduling_information['end_time']

    earliest_end_time = min(finish_times.values())  # Find the minimum value in the dictionary
    return [key for key, value in finish_times.items() if value == earliest_end_time]


def check_precedence_relations(simulationEnv, operation):
    """Check if all precedence relations of an operation are satisfied"""
    for preceding_operation in operation.predecessors:
        if preceding_operation not in simulationEnv.processed_operations:
            return False
    return True


def get_operations_remaining_fast(jobshopEnv, operation):
    """get remaining operations of the job using fast lookup"""
    return len(
        [op for op in operation.job.operations if jobshopEnv._scheduled_operation_indics[op.operation_id] == 0])


def get_work_remaining_fast(jobshopEnv, operation):
    """get amount of work remaining of the job using fast lookup"""
    operations_remaining = [op for op in operation.job.operations if 
                          jobshopEnv._scheduled_operation_indics[op.operation_id] == 0]
    return sum([sum(op.processing_times.values()) / len(op.processing_times) for op in operations_remaining])


def get_earliest_end_time_machines_fast(jobshopEnv, operation):
    """get earliest end time of machines, when operation would be scheduled on it (fast version)"""
    finish_times = {}
    machine_options = operation.processing_times.keys()
    
    for machine_option in machine_options:
        machine = jobshopEnv.get_machine(machine_option)
        if not machine._processed_operations:
            finish_times[machine_option] = jobshopEnv._now + operation.processing_times[machine_option]
        else:
            if jobshopEnv._sequence_dependent_setup_times:
                finish_times[machine_option] = operation.processing_times[machine_option] \
                                           + machine._processed_operations[-1].scheduling_information['end_time'] \
                                           + jobshopEnv._sequence_dependent_setup_times[machine.machine_id][
                                            machine._processed_operations[-1].operation_id][operation.operation_id]
            else:
                finish_times[machine_option] = operation.processing_times[machine_option] \
                                           + machine._processed_operations[-1].scheduling_information['end_time']

    earliest_end_time = min(finish_times.values())
    return [key for key, value in finish_times.items() if value == earliest_end_time]


def check_precedence_relations_fast(jobshopEnv, operation):
    """Check if all precedence relations of an operation are satisfied using fast lookup"""
    for preceding_operation in operation.predecessors:
        if jobshopEnv._scheduled_operation_indics[preceding_operation.operation_id] == 0:
            return False
    return True
