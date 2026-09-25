from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.dispatching_rules.src.rules import *
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.dispatching_rules.src.helper_functions import *
import random

def select_operation_and_machine_fast(jobShopEnv, dispatching_rule='SPT', machine_assignment_rule='SPT'):
    """
    Select next operation and machine based on dispatching rules.
    
    Args:
        jobShopEnv: JobShop environment
        dispatching_rule: Rule for operation selection ('FIFO', 'SPT', 'MOR', 'MWR', 'LOR', 'LWR')
        machine_assignment_rule: Rule for machine selection ('SPT', 'EET')
    
    Returns:
        tuple: (selected_operation, machine_id, duration) or (None, None, None) if no valid operation found
    """
    operation_priorities = {}
    
    # Check each available operation
    for operation in jobShopEnv.operations_available_for_scheduling:
        # Skip if operation is already scheduled
        if operation in jobShopEnv.scheduled_operations:
            continue
            
        # Handle SPT machine assignment rule
        if machine_assignment_rule == 'SPT':

            best_machine = min(operation.processing_times.items(), 
                                key=lambda x: x[1])[0]
            best_duration = operation.processing_times[best_machine]
            
            # Use imported priority functions
            if dispatching_rule == 'FIFO':
                priority = fifo_priority(operation)
            elif dispatching_rule == 'SPT':
                priority = spt_priority(operation)
            elif dispatching_rule == 'MOR':
                priority = mor_priority_fast(jobShopEnv, operation)
            elif dispatching_rule == 'MWR':
                priority = mwr_priority_fast(jobShopEnv, operation)
            elif dispatching_rule == 'LOR':
                priority = lor_priority_fast(jobShopEnv, operation)
            elif dispatching_rule == 'LWR':
                priority = lwr_priority_fast(jobShopEnv, operation)
            else:
                raise ValueError(f"Unknown dispatching rule: {dispatching_rule}")
                
            operation_priorities[(operation, best_machine, best_duration)] = priority
                
        # Handle EET machine assignment rule
        elif machine_assignment_rule == 'EET':
            best_machine = None
            best_end_time = float('inf')
            
            for machine_id in operation.processing_times:
                machine_end_times = [op.scheduling_information['end_time']
                                    for op in jobShopEnv.scheduled_operations
                                    if op.scheduling_information['machine_id'] == machine_id]
                machine_available_time = max(machine_end_times) if machine_end_times else 0
                
                possible_end_time = machine_available_time + operation.processing_times[machine_id]
                if possible_end_time < best_end_time:
                    best_end_time = possible_end_time
                    best_machine = machine_id
                    best_duration = operation.processing_times[machine_id]
            
            # Use imported priority functions
            if dispatching_rule == 'FIFO':
                priority = fifo_priority(operation)
            elif dispatching_rule == 'MOR':
                priority = mor_priority_fast(jobShopEnv, operation)
            elif dispatching_rule == 'MWR':
                priority = mwr_priority_fast(jobShopEnv, operation)
            elif dispatching_rule == 'LOR':
                priority = lor_priority_fast(jobShopEnv, operation)
            elif dispatching_rule == 'LWR':
                priority = lwr_priority_fast(jobShopEnv, operation)
            else:
                raise ValueError(f"Unknown dispatching rule: {dispatching_rule}")
                
            operation_priorities[(operation, best_machine, best_duration)] = priority

    if not operation_priorities:
        return None, None, None
        
    # Select operation based on priority
    if dispatching_rule in ['FIFO', 'SPT', 'LOR', 'LWR']:
        selected = min(operation_priorities, key=operation_priorities.get)
    else:  # MOR, MWR
        selected = max(operation_priorities, key=operation_priorities.get)
        
    return selected  # Returns (operation, machine_id, duration)

def select_operation(simulationEnv, machine, dispatching_rule, machine_assignment_rule):
    """
    Use dispatching rules to select the next operation to schedule.

    Parameters:
    - simulationEnv: The simulation environment containing information on jobs, machines, and operations.
    - machine: The machine object on which an operation is to be scheduled.
    - dispatching_rule: A string representing the rule used to prioritize jobs for scheduling (e.g., 'SPT', 'FIFO').
    - machine_assignment_rule: A string representing the rule for assigning operations to machines (e.g., 'SPT', 'EET').

    Returns:
    - operation_priorities: A dictionary mapping operations to their priority scores based on the given rules.
    """

    # Initialize an empty dictionary to store operations and their computed priorities.
    operation_priorities = {}

    # Iterate over all jobs in the simulation environment
    for job in simulationEnv.jobShopEnv.jobs:
        # Iterate over all operations in the job
        for operation in job.operations:

            # Check if the operation has not been processed or scheduled on a machine
            if operation not in simulationEnv.processed_operations and operation not in simulationEnv.jobShopEnv.scheduled_operations and machine.machine_id in operation.processing_times:
                if check_precedence_relations(simulationEnv, operation):

                    # Check if the machine assignment rule is 'SPT' and the operation has the shortest processing time
                    # on the machine
                    if machine_assignment_rule == 'SPT' and spt_rule(operation, machine.machine_id):
                        if dispatching_rule == 'FIFO':
                            operation_priorities[operation] = fifo_priority(operation)
                        elif dispatching_rule == 'SPT':
                            operation_priorities[operation] = spt_priority(operation)
                        elif dispatching_rule == 'MOR':
                            operation_priorities[operation] = mor_priority(simulationEnv, operation)
                        elif dispatching_rule == 'MWR':
                            operation_priorities[operation] = mwr_priority(simulationEnv, operation)
                        elif dispatching_rule == 'LOR':
                            operation_priorities[operation] = lor_priority(simulationEnv, operation)
                        elif dispatching_rule == 'LWR':
                            operation_priorities[operation] = lwr_priority(simulationEnv, operation)

                    # Check if the machine assignment rule is 'EET' and the operation has the earliest end time
                    # on the machine
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
        if dispatching_rule in ['FIFO', 'SPT', 'LOR', 'LWR']:
            return min(operation_priorities, key=operation_priorities.get)
        else:
            return max(operation_priorities, key=operation_priorities.get)


def schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule):
    """Schedule operations on the machines based on the priority values."""
    machines_available = [
        machine for machine in simulationEnv.jobShopEnv.machines
        if simulationEnv.machine_resources[machine.machine_id].count == 0
    ]
    machines_available.sort(key=lambda m: m.machine_id)
    for machine in machines_available:
        operation_to_schedule = select_operation(simulationEnv, machine, dispatching_rule, machine_assignment_rule)
        if operation_to_schedule is not None:
            simulationEnv.jobShopEnv._scheduled_operations.append(operation_to_schedule)
            simulationEnv.simulator.process(simulationEnv.perform_operation(operation_to_schedule, machine))


def scheduler(simulationEnv, **kwargs):
    """Scheduler for batch mode or online arrivals."""
    dispatching_rule = kwargs['instance']['dispatching_rule']
    machine_assignment_rule = kwargs['instance']['machine_assignment_rule']

    if dispatching_rule == 'SPT' and machine_assignment_rule != 'SPT':
        raise ValueError("SPT dispatching rule requires SPT machine assignment rule.")

    if not simulationEnv.online_arrivals:
        # Add machine resources to the environment
        for _ in simulationEnv.jobShopEnv.machines:
            simulationEnv.add_machine_resources()

        # Run the scheduling environment until all operations are processed
        while len(simulationEnv.processed_operations) < sum(len(job.operations) for job in simulationEnv.jobShopEnv.jobs):
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule)
            yield simulationEnv.simulator.timeout(1)

    else:
        # Start the online job generation process
        simulationEnv.simulator.process(simulationEnv.generate_online_job_arrivals())

        # Run the scheduling environment continuously
        while True:
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule)
            yield simulationEnv.simulator.timeout(1)


def scheduler_step(simulationEnv, **kwargs):
    """Scheduler for batch mode or online arrivals."""
    dispatching_rule = kwargs['instance']['dispatching_rule']
    machine_assignment_rule = kwargs['instance']['machine_assignment_rule']

    if dispatching_rule == 'SPT' and machine_assignment_rule != 'SPT':
        raise ValueError("SPT dispatching rule requires SPT machine assignment rule.")

    if not simulationEnv.online_arrivals:
        # Add machine resources to the environment
        for _ in simulationEnv.jobShopEnv.machines:
            simulationEnv.add_machine_resources()

        # Run the scheduling environment until all operations are processed
        previous_number_of_operations = len(simulationEnv.processed_operations)
        while len(simulationEnv.processed_operations) < sum(len(job.operations) for job in simulationEnv.jobShopEnv.jobs):
            current_number_of_operations = len(simulationEnv.processed_operations)
            if current_number_of_operations > previous_number_of_operations:
                print("decision time:", previous_number_of_operations,current_number_of_operations)
                # dispatching_rule = random.choice(['FIFO', 'MOR', 'MWR', 'LOR', 'LWR'])
                # dispatching_rule = random.choice(['FIFO', 'MOR'])
                print("dispatching rule:", dispatching_rule)
                previous_number_of_operations = current_number_of_operations
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule)
            print("_now",simulationEnv.simulator.now)
            yield simulationEnv.simulator.timeout(1)

    else:
        # Start the online job generation process
        simulationEnv.simulator.process(simulationEnv.generate_online_job_arrivals())

        # Run the scheduling environment continuously
        while True:
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule)
            yield simulationEnv.simulator.timeout(1)



def scheduler_random_dispatching_rule(simulationEnv, **kwargs):
    """Scheduler for batch mode or online arrivals. Randomly select dispatching rule."""

    dispatching_rule = random.choice(['FIFO', 'MOR', 'MWR', 'LOR', 'LWR', 'SPT'])
    if dispatching_rule == 'SPT':
        machine_assignment_rule = 'SPT'
    else:
        machine_assignment_rule = "EET"

    if dispatching_rule == 'SPT' and machine_assignment_rule != 'SPT':
        raise ValueError("SPT dispatching rule requires SPT machine assignment rule.")

    if not simulationEnv.online_arrivals:
        # Add machine resources to the environment
        for _ in simulationEnv.jobShopEnv.machines:
            simulationEnv.add_machine_resources()

        # Run the scheduling environment until all operations are processed
        previous_number_of_operations = len(simulationEnv.processed_operations)
        while len(simulationEnv.processed_operations) < sum(len(job.operations) for job in simulationEnv.jobShopEnv.jobs):
            current_number_of_operations = len(simulationEnv.processed_operations)
            if current_number_of_operations > previous_number_of_operations:
                dispatching_rule = random.choice(['FIFO', 'MOR', 'MWR', 'LOR', 'LWR', 'SPT'])
                if dispatching_rule == 'SPT':
                    machine_assignment_rule = 'SPT'
                else:
                    machine_assignment_rule = "EET"
                # dispatching_rule = random.choice(['FIFO', 'MOR'])
                previous_number_of_operations = current_number_of_operations
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule)
            
            yield simulationEnv.simulator.timeout(1)

    else:
        # Start the online job generation process
        simulationEnv.simulator.process(simulationEnv.generate_online_job_arrivals())

        # Run the scheduling environment continuously
        while True:
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule)
            yield simulationEnv.simulator.timeout(1)