# Code based on the paper:
# "Solving the flexible job shop scheduling problem with sequence-dependent setup times"
# by Liji Shen, Stéphane Dauzère-Pérès, Janis S. Neufeld
# Presented in European Journal of Operational Research, 2018.
# Paper URL: https://www.sciencedirect.com/science/article/pii/S037722171730752X
import re

from gurobipy import GRB, Model, quicksum


def update_env(jobShopEnv, results):
    schedule = {machine: {} for machine in jobShopEnv.machines}
    for var, value in results['variables'].items():
        if 'Y_' in var and value == 1.0:
            numbers = [int(number) for number in re.findall(r'\d+', var)]
            operation = jobShopEnv.get_operation(numbers[1])
            machine = jobShopEnv.get_machine(numbers[2])
            start_time = results['variables']['S_' + str(numbers[0]) + '_' + str(numbers[1])]
            schedule[machine][operation] = start_time

    for machine, operations in schedule.items():
        sorted_operations = sorted(operations, key=lambda k: operations[k])
        for value, operation in enumerate(sorted_operations):
            start_time = operations[operation]
            processing_time = operation.processing_times[machine.machine_id]
            if value == 0:
                setup_time = 0
            else:
                setup_time = jobShopEnv._sequence_dependent_setup_times[machine.machine_id][sorted_operations[value-1].operation_id][operation.operation_id]
            machine.add_operation_to_schedule_at_time(operation, start_time, processing_time, setup_time)
    return jobShopEnv


def fjsp_sdst_milp(jobShopEnv, time_limit):
    # Extracting the instance info from the environment
    jobs = [job.job_id for job in jobShopEnv.jobs]
    operations_per_job = {
        job.job_id: [operation.operation_id for operation in job.operations]
        for job in jobShopEnv.jobs
    }
    machine_allocations = {
        (operation.job_id, operation.operation_id): operation.optional_machines_id
        for operation in jobShopEnv.operations
    }
    operations_times = {
        (
            operation.job_id,
            operation.operation_id,
            operation.optional_machines_id[i],
        ): operation.processing_times[operation.optional_machines_id[i]]
        for operation in jobShopEnv.operations
        for i in range(len(operation.optional_machines_id))
    }
    sdst = {
        (
            operation_i.job_id,
            operation_i.operation_id,
            operation_j.job_id,
            operation_j.operation_id,
            machine_id,
        ): jobShopEnv._sequence_dependent_setup_times[machine_id][
            operation_i.operation_id
        ][
            operation_j.operation_id
        ]
        for machine_id in range(jobShopEnv.nr_of_machines)
        for operation_i in jobShopEnv.operations
        for operation_j in jobShopEnv.operations
    }
    largeM = 10000
    model = Model("FJSP_SDST_MILP")

    # Decision Variables
    Y = {}  # αijk: 1 if Oij is assigned to machine k, 0 otherwise
    S = {}  # Sij for start time of operation Oij
    X = {}  # βiji'j': 1 if Oij is scheduled before Oi'j', 0 otherwise

    for j in jobs:
        for l in operations_per_job[j]:
            S[j, l] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"S_{j}_{l}")
            for i in machine_allocations[j, l]:
                Y[j, l, i] = model.addVar(vtype=GRB.BINARY, name=f"Y_{j}_{l}_{i}")

    for j in jobs:
        for l in operations_per_job[j]:
            for h in jobs:
                for z in operations_per_job[h]:
                    if not (j == h and l == z):
                        X[j, l, h, z] = model.addVar(
                            vtype=GRB.BINARY, name=f"X_{j}_{l}_{h}_{z}"
                        )

    # Objective Function: Minimize Cmax
    cmax = model.addVar(vtype=GRB.CONTINUOUS, name="Cmax")
    model.setObjective(cmax, GRB.MINIMIZE)

    # Constraints (3): Each operation is assigned to one and only one eligible machine
    for j in jobs:
        for l in operations_per_job[j]:
            model.addConstr(
                quicksum(Y[j, l, i] for i in machine_allocations[j, l]) == 1
            )

    # Constraints (4): Precedence relations between consecutive operations of the same job
    for j in jobs:
        for l in operations_per_job[j][1:]:
            model.addConstr(
                S[j, l]
                >= S[j, l - 1]
                + quicksum(
                    operations_times[j, l - 1, i] * Y[j, l - 1, i]
                    for i in machine_allocations[j, l - 1]
                )
            )

    # Constraints (5) & (6): No overlapping of operations on the same machine k
    for j in jobs:
        for l in operations_per_job[j]:
            for h in jobs:
                for z in operations_per_job[h]:
                    if not (j == h and l == z):
                        common_machines = set(machine_allocations[j, l]) & set(
                            machine_allocations[h, z]
                        )
                        for i in common_machines:
                            model.addConstr(
                                S[j, l]
                                >= S[h, z]
                                + operations_times[h, z, i]
                                + sdst[h, z, j, l, i]
                                - (
                                    (2 - Y[j, l, i] - Y[h, z, i] + X[j, l, h, z])
                                    * largeM
                                )
                            )

    for j in jobs:
        for l in operations_per_job[j]:
            for h in jobs:
                for z in operations_per_job[h]:
                    if not (j == h and l == z):
                        common_machines = set(machine_allocations[j, l]) & set(
                            machine_allocations[h, z]
                        )
                        for i in common_machines:
                            model.addConstr(
                                S[h, z]
                                >= S[j, l]
                                + operations_times[j, l, i]
                                + sdst[j, l, h, z, i]
                                - (
                                    (3 - Y[j, l, i] - Y[h, z, i] - X[j, l, h, z])
                                    * largeM
                                )
                            )

    # Constraints (7): Determine makespan
    for j in jobs:
        last_op = max(operations_per_job[j])
        model.addConstr(
            cmax
            >= S[j, last_op]
            + quicksum(
                operations_times[j, last_op, i] * Y[j, last_op, i]
                for i in machine_allocations[j, last_op]
            )
        )

    model.params.TimeLimit = time_limit

    return model
