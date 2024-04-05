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
        lines = data.readlines()

        total_jobs, total_machines, max_operations = lines[0].split()
        number_total_jobs, number_total_machines, number_max_operations = int(
            total_jobs), int(total_machines), int(float(max_operations))

        JobShop.set_nr_of_jobs(number_total_jobs)
        JobShop.set_nr_of_machines(number_total_machines)

        precedence_relations = {}
        job_id = 0
        operation_id = 0

        for key, line in enumerate(lines[1:]):
            if key < number_total_jobs:
                # Split data with multiple spaces as separator
                parsed_line = line.split()

                # Current item of the parsed line
                i = 1
                job = Job(job_id)

                while i < len(parsed_line):
                    # Total number of operation options for the operation
                    operation_options = int(parsed_line[i])
                    # Current activity
                    operation = Operation(job, job_id, operation_id)
                    for operation_options_id in range(operation_options):
                        operation.add_operation_option(int(parsed_line[i + 2 * operation_options_id + 1])-1,
                                                                       int(parsed_line[i + 2 + 2 * operation_options_id]))

                    job.add_operation(operation)
                    JobShop.add_operation(operation)
                    if i != 1:
                        precedence_relations[operation_id] = [
                            JobShop.get_operation(operation_id - 1)]

                    i += 1 + 2 * operation_options
                    operation_id += 1

                JobShop.add_job(job)
                job_id += 1
            else:
                break
                # For each machine and each operation: Setup time between the first operation and every other
                # operation on the machine considered

        counter_machine_id = 0
        counter_operation_id = 0
        sequence_dependent_setup_times = [[[-1 for r in range(len(JobShop.operations))] for t in range(
            len(JobShop.operations))] for m in range(number_total_machines)]

        for line in lines[number_total_jobs + 2:]:
            sequence_dependent_setup_times[counter_machine_id][counter_operation_id] = list(
                map(int, line.split()))
            counter_operation_id += 1
            if counter_operation_id == len(JobShop.operations):
                counter_machine_id += 1
                counter_operation_id = 0

    # add also the operations without precedence operations to the precendence relations dictionary
    for operation in JobShop.operations:
        if operation.operation_id not in precedence_relations.keys():
            precedence_relations[operation.operation_id] = []
        operation.add_predecessors(
            precedence_relations[operation.operation_id])

    # Precedence Relations
    JobShop.add_precedence_relations_operations(precedence_relations)
    JobShop.add_sequence_dependent_setup_times(sequence_dependent_setup_times)

    # Machines
    for id_machine in range(0, number_total_machines):
        JobShop.add_machine((Machine(id_machine)))

    return JobShop
