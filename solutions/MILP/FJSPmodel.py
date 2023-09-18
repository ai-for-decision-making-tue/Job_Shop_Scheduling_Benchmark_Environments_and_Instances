from gurobipy import Model, GRB, quicksum


def parse_file(filename):
    # Initialize variables
    machine_allocations = {}
    operations_times = {}
    sdsts = {}
    numonJobs = []
    total_op_nr = 0

    #TODO
    with open("./data/" + filename, 'r') as f:
        # Extract header data
        number_operations, number_machines, _ = map(float, f.readline().split())
        number_jobs = int(number_operations)
        number_machines = int(number_machines)

        # Process job operations data
        for i in range(number_jobs):
            operation_data = list(map(int, f.readline().split()))
            operation_num = operation_data[0]
            numonJobs.append(operation_num)

            index, operation_id = 1, 0
            while index < len(operation_data):
                print(operation_id)
                total_op_nr += 1

                # Extract machine and processing time data for each operation
                o_num = operation_data[index]
                job_machines = operation_data[index + 1:index + 1 + o_num * 2:2]
                job_processingtime = operation_data[index + 2:index + 2 + o_num * 2:2]
                machine_allocations[(i + 1, operation_id + 1)] = job_machines

                # Save processing times
                for l, machine in enumerate(job_machines):
                    operations_times[(i + 1, operation_id + 1, machine)] = job_processingtime[l]

                operation_id += 1
                index += o_num * 2 + 1

        # Process sdsts data
        machine_id, operation_id = 1, 1
        total_ops = len(machine_allocations.keys())
        for line in f:
            sdst_values = list(map(int, line.split()))

            for ix, sdst in enumerate(sdst_values):
                sdsts[(machine_id, operation_id, ix + 1)] = sdst

            if operation_id == total_ops:
                operation_id = 0
                machine_id += 1
            operation_id += 1

    # Calculate derived values
    jobs = list(range(1, number_jobs + 1))
    machines = list(range(1, number_machines + 1))
    operations_per_job = {j: list(range(1, numonJobs[j - 1] + 1)) for j in jobs}
    largeM = sum(
        max(operations_times[(job, op, l)] for l in machine_allocations[(job, op)]) for job in jobs for op in
        operations_per_job[job]
    )

    # Return parsed data
    return {
        'number_jobs': number_jobs,
        'number_machines': number_machines,
        'jobs': jobs,
        'machines': machines,
        'operations_per_job': operations_per_job,
        'machine_allocations': machine_allocations,
        'operations_times': operations_times,
        'largeM': largeM,  #
        "sdsts": sdsts
    }


def fjsp_milp(Data):
    # Extracting the data
    jobs = Data['jobs']  # j,h
    operations_per_job = Data['operations_per_job']  # l,z
    machine_allocations = Data['machine_allocations']  # Rj,l
    operations_times = Data['operations_times']  # pj,l,i
    largeM = Data['largeM']  # M
    model = Model("MILP-5")

    # Decision Variables
    Y = {}
    S = {}
    X = {}

    for j in jobs:
        for l in operations_per_job[j]:
            for i in machine_allocations[j, l]:
                S[j, l, i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"S_{j}_{l}_{i}")
                Y[j, l, i] = model.addVar(vtype=GRB.BINARY, name=f"Y_{j}_{l}_{i}")

                for h in jobs:
                    if h > j:
                        for z in operations_per_job[h]:
                            X[j, l, h, z] = model.addVar(vtype=GRB.BINARY, name=f"X_{j}_{l}_{h}_{z}")

    # Objective function
    cmax = model.addVar(vtype=GRB.CONTINUOUS, name="Cmax")
    model.setObjective(cmax, GRB.MINIMIZE)

    # Constraints

    # 3.2.2 Assignment constraint sets
    for j in jobs:
        for l in operations_per_job[j]:
            model.addConstr(quicksum(Y[j, l, i] for i in machine_allocations[j, l]) == 1)
            for i in machine_allocations[j, l]:
                model.addConstr(S[j, l, i] <= largeM * Y[j, l, i])

    # 3.2.3.1 Logical precedence constraints among operations of a job
    for j in jobs:
        for l in operations_per_job[j][:-1]:
            lhs = quicksum(S[j, l + 1, i] for i in machine_allocations[j, l + 1])
            rhs = quicksum(S[j, l, i] for i in machine_allocations[j, l]) + quicksum(
                operations_times[j, l, i] * Y[j, l, i] for i in machine_allocations[j, l])
            model.addConstr(lhs >= rhs)

    # 3.2.3.2 Machine non-interference constraints and precedence constraints among operations of different jobs
    for j in jobs:
        for l in operations_per_job[j]:
            for h in jobs:
                if h > j:
                    for z in operations_per_job[h]:
                        common_machines = set(machine_allocations[j, l]) & set(machine_allocations[h, z])
                        for i in common_machines:
                            model.addConstr(
                                S[j, l, i] >= S[h, z, i] + operations_times[h, z, i] - (largeM * (
                                        3 - X[j, l, h, z] - Y[j, l, i] - Y[h, z, i])))
                            model.addConstr(
                                S[h, z, i] >= S[j, l, i] + operations_times[j, l, i] - (largeM * (
                                        X[j, l, h, z] + 2 - Y[j, l, i] - Y[h, z, i])))

    # 3.2.3.3 Constraints for capturing the value of objective function
    for j in jobs:
        last_op = max(operations_per_job[j])
        model.addConstr(
            cmax >= quicksum(S[j, last_op, k] for k in machine_allocations[j, last_op]) + quicksum(
                operations_times[j, last_op, k] * (Y[j, last_op, k]) for k in machine_allocations[j, last_op])
        )

    # model.params.TimeLimit = 1

    return model
