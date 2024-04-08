# Code based on the paper: "On the job-shop scheduling problem"
# by A. S. Manne, Presented in Operations Research, 1960.
# Paper URL: https://www.jstor.org/stable/167204
import re

from gurobipy import GRB, Model


def update_env(jobShopEnv, results):
    for var, start_time in results['variables'].items():
        if 'x_' in var:
            numbers = [int(number) for number in re.findall(r'\d+', var)]
            operation = jobShopEnv.get_operation(numbers[1])
            machine = jobShopEnv.get_machine(numbers[2])
            processing_time = operation.processing_times[machine.machine_id]
            setup_time = 0
            machine.add_operation_to_schedule_at_time(operation, start_time, processing_time, setup_time)
    return jobShopEnv


def jsp_milp(jobShopEnv, time_limit):
    # Extracting the instance info from the environment
    jobs = [job.job_id for job in jobShopEnv.jobs]
    operations_per_job = {
        job.job_id: [operation.operation_id for operation in job.operations]
        for job in jobShopEnv.jobs
    }
    machine_allocations = {
        (operation.job_id, operation.operation_id): operation.optional_machines_id[0]
        for operation in jobShopEnv.operations
    }
    operations_times = {
        (
            operation.job_id,
            operation.operation_id,
            operation.optional_machines_id[0],
        ): operation.processing_times[operation.optional_machines_id[0]]
        for operation in jobShopEnv.operations
    }
    largeM = sum(
        [processing_time for operation, processing_time in operations_times.items()]
    )
    model = Model("JSP_MILP")

    # Decision variables
    x = {}
    z = {}

    for j in jobs:
        for l in operations_per_job[j]:
            i = machine_allocations[(j, l)]
            x[(j, l, i)] = model.addVar(vtype=GRB.INTEGER, name=f"x_{j}_{l}_{i}", lb=0)
            for h in jobs:
                if h > j:
                    for k in operations_per_job[h]:
                        if machine_allocations[(h, k)] == i:
                            z[(j, l, i, h, k)] = model.addVar(
                                vtype=GRB.BINARY, name=f"z_{j}_{l}_{i}_{h}_{k}"
                            )

    # Objective function (1)
    cmax = model.addVar(vtype=GRB.CONTINUOUS, name="Cmax")
    model.setObjective(cmax, GRB.MINIMIZE)

    # Constraints

    # Operation start time constraints (2)
    for j in jobs:
        for l in operations_per_job[j]:
            model.addConstr(x[(j, l, machine_allocations[(j, l)])] >= 0)

    # Precedence constraints (3)
    for j in jobs:
        for l in operations_per_job[j][:-1]:
            model.addConstr(
                x[(j, l + 1, machine_allocations[(j, l + 1)])]
                >= x[(j, l, machine_allocations[(j, l)])]
                + operations_times[(j, l, machine_allocations[(j, l)])]
            )

    # Disjunctive constraints (no two jobs can be scheduled on the same machine at the same time) (4 & 5)
    for j in jobs:
        for l in operations_per_job[j]:
            for h in jobs:
                if h > j:
                    for k in operations_per_job[h]:
                        if machine_allocations[(h, k)] == machine_allocations[(j, l)]:
                            model.addConstr(
                                x[(h, k, machine_allocations[(h, k)])]
                                + operations_times[(h, k, machine_allocations[(h, k)])]
                                <= x[(j, l, machine_allocations[(j, l)])]
                                + largeM * z[(j, l, machine_allocations[(j, l)], h, k)]
                            )
                            model.addConstr(
                                x[(j, l, machine_allocations[(j, l)])]
                                + operations_times[(j, l, machine_allocations[(j, l)])]
                                <= x[(h, k, machine_allocations[(h, k)])]
                                + largeM
                                * (1 - z[(j, l, machine_allocations[(j, l)], h, k)])
                            )

    # Capture objective function (6)
    for j in jobs:
        model.addConstr(
            cmax
            >= x[
                j,
                max(operations_per_job[j]),
                machine_allocations[j, max(operations_per_job[j])],
            ]
            + operations_times[
                j,
                max(operations_per_job[j]),
                machine_allocations[j, max(operations_per_job[j])],
            ]
        )

    model.params.TimeLimit = time_limit

    return model
