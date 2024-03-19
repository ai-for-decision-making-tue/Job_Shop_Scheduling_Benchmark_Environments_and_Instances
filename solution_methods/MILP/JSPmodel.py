# Code based on the paper: "On the job-shop scheduling problem"
# by A. S. Manne, Presented in Operations Research, 1960.
# Paper URL: https://www.jstor.org/stable/167204

from gurobipy import Model, GRB


def parse_file(filename):
    # Initialize variables
    machine_allocations = {}
    operations_times = {}
    total_op_nr = 0

    with open("./data/" + filename, 'r') as f:
        # Extract header data
        number_operations, number_machines = map(float, f.readline().split())
        number_jobs = int(number_operations)
        number_machines = int(number_machines)
        operations_per_job = {j: [] for j in range(1, number_jobs + 1)}

        # Process job operations data
        for i in range(number_jobs):
            operation_data = list(map(int, f.readline().split()))
            index, operation_id = 0, 0
            while index < len(operation_data):
                total_op_nr += 1

                job_machines = operation_data[index]
                job_processingtime = operation_data[index + 1]
                machine_allocations[(i + 1, operation_id + 1)] = job_machines
                operations_times[(i + 1, operation_id + 1, job_machines)] = job_processingtime
                operations_per_job[i+1].append(operation_id + 1)

                operation_id += 1
                index += 2

    # Calculate derived values
    jobs = list(range(1, number_jobs + 1))
    machines = list(range(0, number_machines))
    # calculate upper bound
    largeM = sum([processing_time for operation,processing_time in operations_times.items()])

    # Return parsed data
    return {
        'number_jobs': number_jobs,
        'number_machines': number_machines,
        'jobs': jobs,
        'machines': machines,
        'operations_per_job': operations_per_job,
        'machine_allocations': machine_allocations,
        'operations_times': operations_times,
        'largeM': largeM,
    }


def jsp_milp(instance_data, time_limit):
    # Extracting the data
    jobs = instance_data['jobs']
    operations_per_job = instance_data['operations_per_job']
    machine_allocations = instance_data['machine_allocations']
    operations_times = instance_data['operations_times']
    largeM = instance_data['largeM']
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
                            z[(j, l, i, h, k)] = model.addVar(vtype=GRB.BINARY, name=f"z_{j}_{l}_{i}_{h}_{k}")

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
            model.addConstr(x[(j, l+1, machine_allocations[(j, l+1)])] >=
                            x[(j, l, machine_allocations[(j, l)])] + operations_times[(j, l, machine_allocations[(j, l)])])

    # Disjunctive constraints (no two jobs can be scheduled on the same machine at the same time) (4 & 5)
    for j in jobs:
        for l in operations_per_job[j]:
            for h in jobs:
                if h > j:
                    for k in operations_per_job[h]:
                        if machine_allocations[(h, k)] == machine_allocations[(j, l)]:
                            model.addConstr(x[(h, k, machine_allocations[(h, k)])] + operations_times[(h, k, machine_allocations[(h, k)])] <=
                                            x[(j, l, machine_allocations[(j, l)])] + largeM * z[(j, l, machine_allocations[(j, l)], h, k)])
                            model.addConstr(x[(j, l, machine_allocations[(j, l)])] + operations_times[(j, l, machine_allocations[(j, l)])] <=
                                            x[(h, k, machine_allocations[(h, k)])] + largeM * (1 - z[(j, l, machine_allocations[(j, l)], h, k)]))

    # Capture objective function (6)
    for j in jobs:
        model.addConstr(
            cmax >= x[j, max(operations_per_job[j]), machine_allocations[j, max(operations_per_job[j])]] +
            operations_times[j, max(operations_per_job[j]), machine_allocations[j, max(operations_per_job[j])]])

    model.params.TimeLimit = time_limit

    return model