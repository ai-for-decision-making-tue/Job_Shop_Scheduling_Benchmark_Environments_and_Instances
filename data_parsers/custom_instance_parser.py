from scheduling_environment.job import Job
from scheduling_environment.machine import Machine
from scheduling_environment.operation import Operation
from scheduling_environment.jobShop import JobShop


def parse(processing_info, instance_name="custom_problem_instance"):

    # Initialize JobShop
    jobShop = JobShop()
    jobShop.set_instance_name(instance_name)

    # Configure Machines based on nr_machines in processing_info
    number_total_machines = processing_info["nr_machines"]
    for machine_id in range(0, number_total_machines):
        jobShop.add_machine(Machine(machine_id))
    jobShop.set_nr_of_machines(number_total_machines)

    # Configure jobs, operations, and processing times
    for job_info in processing_info["jobs"]:
        job = Job(job_id=job_info["job_id"])

        for operation_info in job_info["operations"]:
            operation = Operation(job, job_info["job_id"], operation_info["operation_id"])

            # Convert machine names (e.g., "machine_1") to numeric IDs for compatibility
            for machine_key, processing_time in operation_info["processing_times"].items():
                machine_id = int(machine_key.split("_")[1])-1
                operation.add_operation_option(machine_id, processing_time)

            job.add_operation(operation)
            jobShop.add_operation(operation)
        jobShop.add_job(job)
    jobShop.set_nr_of_jobs(len(processing_info["jobs"]))

    # Configure precedence relations between operations
    precedence_relations = {}
    for job_info in processing_info["jobs"]:
        for op_info in job_info["operations"]:
            if op_info["predecessors"] is not None:
                operation = jobShop.get_operation(op_info["operation_id"])
                precedence_relations[op_info["operation_id"]] = []
                for predecessor in op_info["predecessors"]:
                    predecessor_operation = jobShop.get_operation(predecessor)
                    operation.add_predecessors([predecessor_operation])
                    precedence_relations[op_info["operation_id"]].append(predecessor_operation)
            else:
                precedence_relations[op_info["operation_id"]] = []

    # Configure sequence-dependent setup times for each machine and operation pair
    setup_times = processing_info["sequence_dependent_setup_times"]
    sequence_dependent_setup_times = {}

    for machine_key, setup_matrix in setup_times.items():
        machine_id = int(machine_key.split("_")[1])-1  # Convert machine_1 to machine ID 1
        machine_setup_times = {}

        # Map the setup times for all pairs of operations
        for i in range(len(setup_matrix)):
            for j in range(len(setup_matrix[i])):
                if i != j:  # Ignore setup times for the same operation (i == j)
                    if i not in machine_setup_times:
                        machine_setup_times[i] = {}
                    machine_setup_times[i][j] = setup_matrix[i][j]

        sequence_dependent_setup_times[machine_id] = machine_setup_times

    # Add the precedence relations and sequence-dependent setup times to the JobShop
    jobShop.add_precedence_relations_operations(precedence_relations)
    jobShop.add_sequence_dependent_setup_times(sequence_dependent_setup_times)

    return jobShop


if __name__ == "__main__":
    processing_info = {
        "instance_name": "custom_problem_instance",
        "nr_machines": 2,
        "jobs": [
            {"job_id": 0, "operations": [
                {"operation_id": 0, "processing_times": {"machine_1": 10, "machine_2": 20}, "predecessors": None},
                {"operation_id": 1, "processing_times": {"machine_1": 25, "machine_2": 19}, "predecessors": [0]}
            ]},
            {"job_id": 1, "operations": [
                {"operation_id": 2, "processing_times": {"machine_1": 23, "machine_2": 21}, "predecessors": None},
                {"operation_id": 3, "processing_times": {"machine_1": 12, "machine_2": 24}, "predecessors": [2]}
            ]},
            {"job_id": 2, "operations": [
                {"operation_id": 4, "processing_times": {"machine_1": 37, "machine_2": 21}, "predecessors": None},
                {"operation_id": 5, "processing_times": {"machine_1": 23, "machine_2": 34}, "predecessors": [4]}
            ]}
        ],
        "sequence_dependent_setup_times": {
            "machine_1": [
                [0, 25, 30, 35, 40, 45],
                [25, 0, 20, 30, 40, 50],
                [30, 20, 0, 10, 15, 25],
                [35, 30, 10, 0, 5, 10],
                [40, 40, 15, 5, 0, 20],
                [45, 50, 25, 10, 20, 0]
            ],
            "machine_2": [
                [0, 21, 30, 35, 40, 45],
                [21, 0, 10, 25, 30, 40],
                [30, 10, 0, 5, 15, 25],
                [35, 25, 5, 0, 10, 20],
                [40, 30, 15, 10, 0, 25],
                [45, 40, 25, 20, 25, 0]
            ]
        }
    }

    jobShopEnv = parse(processing_info)
    print('Job shop setup complete')

    # TEST GA:
    # from solution_methods.GA.src.initialization import initialize_run
    # from solution_methods.GA.run_GA import run_GA
    # import multiprocessing
    #
    # parameters = {"instance": {"problem_instance": "custom_problem_instance"},
    #              "algorithm": {"population_size": 8, "ngen": 10, "seed": 5, "indpb": 0.2, "cr": 0.7, "mutiprocessing": True},
    #              "output": {"logbook": True}
    #             }
    #
    # pool = multiprocessing.Pool()
    # population, toolbox, stats, hof = initialize_run(jobShopEnv, pool, **parameters)
    # makespan, jobShopEnv = run_GA(jobShopEnv, population, toolbox, stats, hof, **parameters)

    # TEST CP_SAT:
    from solution_methods.CP_SAT.run_cp_sat import run_CP_SAT
    parameters = {"instance": {"problem_instance": "custom_fjsp_sdst"},
                  "solver": {"time_limit": 3600},
                  "output": {"logbook": True}
                  }

    jobShopEnv = parse(processing_info)
    results, jobShopEnv = run_CP_SAT(jobShopEnv, **parameters)