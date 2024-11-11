import random
import numpy as np

from scheduling_environment.jobShop import JobShop


def random_scheduler(jobShop: JobShop) -> JobShop:
    """Randomly assign jobs to machines.

    :param env: The environment where jobs need to be.
    :return: The environment after jobs have been assigned.
    """

    jobShop.update_operations_available_for_scheduling()
    while len(jobShop.operations_to_be_scheduled) > 0:
        operation = random.choice(jobShop.operations_available_for_scheduling)
        machine_id = random.choice(list(operation.processing_times.keys()))
        duration = operation.processing_times[machine_id]
        jobShop.schedule_operation_with_backfilling(operation, machine_id, duration)
        jobShop.update_operations_available_for_scheduling()
    return jobShop


def greedy_scheduler(jobShop: JobShop) -> JobShop:
    """Greedy assign operations to machines based on shortest processing times

    :param env: The environment where jobs need to be.
    :return: The environment after jobs have been assigned.
    """

    jobShop.update_operations_available_for_scheduling()
    while len(jobShop.operations_to_be_scheduled) > 0:
        best_operation = None
        best_machine_id = None
        best_duration = np.inf
        for operation in jobShop.operations_available_for_scheduling:
            machine_id = min(operation.processing_times, key=lambda k: operation.processing_times[k])
            duration = operation.processing_times[machine_id]
            if duration < best_duration:
                best_operation = operation
                best_machine_id = machine_id
                best_duration = duration
        jobShop.schedule_operation_with_backfilling(best_operation, best_machine_id, best_duration)
        jobShop.update_operations_available_for_scheduling()
    return jobShop


def local_load_balancing_scheduler(jobShop: JobShop) -> JobShop:
    """ local load balancing scheduler

    :param env: The environment where jobs need to be.
    :return: The environment after jobs have been assigned.
    """
    jobs_to_be_scheduled = [job_id for job_id in range(jobShop.nr_of_jobs)]
    jobShop.update_operations_available_for_scheduling()

    while jobs_to_be_scheduled != []:
        jobs_available_for_scheduling = list(
            set(operation.job for operation in jobShop.operations_available_for_scheduling))
        job = random.choice(jobs_available_for_scheduling)
        machine_occupation_times = {machine.machine_id: 0 for machine in jobShop.machines}
        for operation in job.operations:
            updated_occupation_times = {}

            for machine_id in operation.processing_times.keys():
                if machine_id not in updated_occupation_times:
                    updated_occupation_times[machine_id] = operation.processing_times[machine_id]
                else:
                    updated_occupation_times[machine_id] += operation.processing_times[machine_id]

            machine_id = min(updated_occupation_times.items(), key=lambda x: x[1])[0]
            duration = operation.processing_times[machine_id]
            jobShop.schedule_operation_with_backfilling(operation, machine_id, duration)

            machine_occupation_times[machine_id] += duration
            jobShop.update_operations_available_for_scheduling()

        jobs_to_be_scheduled.remove(job.job_id)

    return jobShop


def global_load_balancing_scheduler(jobShop: JobShop) -> JobShop:
    """ global load balancing scheduler

    :param env: The environment where jobs need to be.
    :return: The environment after jobs have been assigned.
    """
    jobs_to_be_scheduled = [job_id for job_id in range(jobShop.nr_of_jobs)]
    machine_occupation_times = {machine.machine_id: 0 for machine in jobShop.machines}
    jobShop.update_operations_available_for_scheduling()

    while jobs_to_be_scheduled != []:
        jobs_available_for_scheduling = list(
            set(operation.job for operation in jobShop.operations_available_for_scheduling))
        job = random.choice(jobs_available_for_scheduling)
        for operation in job.operations:
            updated_occupation_times = {}
            for machine_id in operation.processing_times.keys():
                if machine_id not in updated_occupation_times:
                    updated_occupation_times[machine_id] = operation.processing_times[machine_id]
                else:
                    updated_occupation_times[machine_id] += operation.processing_times[machine_id]

            machine_id = min(updated_occupation_times.items(), key=lambda x: x[1])[0]
            duration = operation.processing_times[machine_id]
            jobShop.schedule_operation_with_backfilling(operation, machine_id, duration)

            machine_occupation_times[machine_id] += duration
            jobShop.update_operations_available_for_scheduling()

        jobs_to_be_scheduled.remove(job.job_id)

    return jobShop
