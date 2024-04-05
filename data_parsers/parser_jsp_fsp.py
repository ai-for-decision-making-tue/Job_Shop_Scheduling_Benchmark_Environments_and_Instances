import re
from pathlib import Path

from scheduling_environment.job import Job
from scheduling_environment.machine import Machine
from scheduling_environment.operation import Operation


def parse(JobShop, instance, from_absolute_path=False):
    if not from_absolute_path:
        base_path = Path(__file__).parent.parent.absolute()
        data_path = base_path.joinpath('data' + instance)
    else:
        data_path = instance

    with open(data_path, "r") as data:
        total_jobs, total_machines = re.findall('\S+', data.readline())
        number_total_jobs, number_total_machines = int(
            total_jobs), int(total_machines)

        JobShop.set_nr_of_jobs(number_total_jobs)
        JobShop.set_nr_of_machines(number_total_machines)
        precedence_relations = {}
        job_id = 0
        operation_id = 0

        for key, line in enumerate(data):
            if key >= number_total_jobs:
                break
            # Split data with multiple spaces as separator
            parsed_line = re.findall('\S+', line)

            # Current item of the parsed line
            i = 0

            job = Job(job_id)

            while i < len(parsed_line):
                # Current operation
                operation = Operation(job, job_id, operation_id)
                operation_options = 1
                for operation_option_id in range(operation_options):
                    operation.add_operation_option(int(parsed_line[i]), int(parsed_line[i + 1]))
                job.add_operation(operation)
                JobShop.add_operation(operation)
                if i != 0:
                    precedence_relations[operation_id] = [
                        JobShop.get_operation(operation_id - 1)]
                i += 2
                operation_id += 1

            JobShop.add_job(job)
            job_id += 1

    # Precedence Relations
    for operation in JobShop.operations:
        if operation.operation_id not in precedence_relations.keys():
            precedence_relations[operation.operation_id] = []
        operation.add_predecessors(
            precedence_relations[operation.operation_id])

    sequence_dependent_setup_times = [[[0 for r in range(len(JobShop.operations))] for t in range(len(JobShop.operations))] for
                                      m in range(number_total_machines)]

    # Precedence Relations
    JobShop.add_precedence_relations_operations(precedence_relations)
    JobShop.add_sequence_dependent_setup_times(sequence_dependent_setup_times)

    # Machines
    for id_machine in range(0, number_total_machines):
        JobShop.add_machine((Machine(id_machine)))

    return JobShop
