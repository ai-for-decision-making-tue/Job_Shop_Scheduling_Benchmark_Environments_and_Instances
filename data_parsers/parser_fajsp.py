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
        total_operations, total_precedence_relations, total_machines = re.findall(
            '\S+', data.readline())
        number_total_operations, number_precedence_relations, number_total_machines = int(
            total_operations), int(total_precedence_relations), int(total_machines)

        precedence_relations = {}
        job_id = 100000

        for key, line in enumerate(data):
            # Split data with multiple spaces as separator
            parsed_line = re.findall('\S+', line)

            if key <= number_precedence_relations - 1:
                if int(parsed_line[1]) in precedence_relations.keys():
                    precedence_relations[int(
                        parsed_line[1])].append(int(parsed_line[0]))
                else:
                    precedence_relations[int(
                        parsed_line[1])] = [int(parsed_line[0])]

            else:
                # Current item of the parsed line
                i = 0
                while i < len(parsed_line):
                    # Total number of operations options for the operation
                    # number of possible options for the operation
                    operation_options = int(parsed_line[0])
                    operation_id = key - number_precedence_relations
                    operation = Operation(None, job_id, operation_id)
                    for operation_option_id in range(operation_options):
                        operation.add_operation_option(int(
                            parsed_line[i + 2 * (operation_option_id + 1) - 1]),
                                                                       int(parsed_line[i + (2 * operation_option_id)+2]))

                    JobShop.add_operation(operation)
                    i += 1 + 2 * operation_options

    # add also the operations without precedence operation to the precendence relations dictionary
    for operation in JobShop.operations:
        if operation.operation_id not in precedence_relations.keys():
            precedence_relations[operation.operation_id] = []
        else:
            precedence_relations[operation.operation_id] = [JobShop.get_operation(
                operation_id) for operation_id in precedence_relations[operation.operation_id]]
        operation.add_predecessors(
            precedence_relations[operation.operation_id])

    sequence_dependent_setup_times = [[[0 for r in range(len(JobShop.operations))] for t in range(len(JobShop.operations))] for
                                      m in range(number_total_machines)]

    # Create artificial job_ids
    job_id = 0
    for operation in JobShop.operations:
        # if it has no predecessors, then it is a starting operation of a new job
        if precedence_relations[operation.operation_id] == []:
            operation.update_job_id(job_id)
            job = Job(job_id)
            JobShop.add_job(job)
            job_id += 1

    for operation in JobShop.operations:
        if operation.job_id == 100000:
            if sum(predecessors.count(operation) for predecessors in precedence_relations.values()) > 1 or len(
                    precedence_relations[operation.operation_id]) > 1:
                operation.update_job_id(job_id)
                job = Job(job_id)
                JobShop.add_job(job)
                job_id += 1

            elif sum(predecessors.count(operation) for predecessors in precedence_relations.values()) == 1 or len(
                    precedence_relations[operation.operation_id]) == 1:
                if sum(predecessors.count(precedence_relations[operation.operation_id][0]) for predecessors in
                       precedence_relations.values()) > 1:
                    operation.update_job_id(job_id)
                    job = Job(job_id)
                    JobShop.add_job(job)
                    job_id += 1

                else:
                    predecessor_job_id = precedence_relations[operation.operation_id][0].job_id
                    operation.update_job_id(predecessor_job_id)

            elif sum(predecessors.count(operation) for predecessors in precedence_relations.values()) == 0 and len(
                    precedence_relations[operation.operation_id]) == 1:
                predecessor_job_id = precedence_relations[operation.operation_id][0].job_id
                operation.update_job_id(predecessor_job_id)

    for operation in JobShop.operations:
        job = JobShop.get_job(operation.job_id)
        job.add_operation(operation)
        operation.update_job(job)

    precedence_relations_jobs = {}
    for operation in JobShop.operations:
        if operation.job_id not in precedence_relations_jobs.keys():
            precedence_relations_jobs[operation.job_id] = [prec.job_id for prec in
                                                           precedence_relations[operation.operation_id]]
        else:
            precedence_relations_jobs[operation.job_id].extend(
                [prec.job_id for prec in precedence_relations[operation.operation_id]])

    # Removing duplicates and values equal to keys
    for key, values in precedence_relations_jobs.items():
        # Remove duplicates from values
        values = list(set(values))

        # Remove values that are equal to the key
        values = [value for value in values if value != key]

        precedence_relations_jobs[key] = values

    JobShop.add_precedence_relations_jobs(precedence_relations_jobs)

    # Precedence Relations & sequence dependent setup times
    JobShop.add_precedence_relations_operations(precedence_relations)
    JobShop.add_sequence_dependent_setup_times(sequence_dependent_setup_times)

    JobShop.set_nr_of_jobs(len(JobShop.jobs))
    JobShop.set_nr_of_machines(number_total_machines)

    # Machines
    for machine_id in range(0, number_total_machines):
        JobShop.add_machine((Machine(machine_id)))

    return JobShop
