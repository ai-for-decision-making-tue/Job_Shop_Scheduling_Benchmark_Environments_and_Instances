from gurobipy import Model, GRB, quicksum


def parse_file(filename):
    # Initialize variables
    machine_allocations = {}
    operations_times = {}
    sdsts = {}
    numonJobs = []
    total_op_nr = 0

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

        job_operation_pairs = list(machine_allocations.keys())
        from_job, from_op = 1, 1
        operation_counter = 0  # This counter will help reset after processing four operations on a machine
        machine_id = 1  # Starting machine ID

        for line in f:
            sdst_values = list(map(int, line.split()))
            # Skip empty lines
            if line != '\n':
                for ix, sdst in enumerate(sdst_values):
                    to_job, to_op = job_operation_pairs[ix]
                    sdsts[(from_job, from_op, to_job, to_op, machine_id)] = sdst

                operation_counter += 1

                # Reset counter if you've processed all operations for the current job on the current machine
                if operation_counter == sum(numonJobs):
                    machine_id += 1
                    from_job = 1
                    from_op = 1
                    operation_counter = 0
                elif from_op == numonJobs[from_job-1]:
                    from_job += 1  # Move to the next job
                    from_op = 1  # Reset operation ID
                else:
                    from_op += 1  # Move to the next operation

    # Calculate derived values
    jobs = list(range(1, number_jobs + 1))
    machines = list(range(1, number_machines + 1))
    operations_per_job = {j: list(range(1, numonJobs[j - 1] + 1)) for j in jobs}

    # Return parsed data
    return {
        'number_jobs': number_jobs,
        'number_machines': number_machines,
        'jobs': jobs,
        'machines': machines,
        'operations_per_job': operations_per_job,
        'machine_allocations': machine_allocations,
        'operations_times': operations_times,
        'largeM': 10000,
        "sdsts": sdsts
    }


def fjsp_sdst_milp(Data, time_limit):
    # Extracting the data
    jobs = Data['jobs']  # j,h
    operations_per_job = Data['operations_per_job']  # l,z
    machine_allocations = Data['machine_allocations']  # Rj,l
    operations_times = Data['operations_times']  # pj,l,i
    largeM = Data['largeM']  # M
    model = Model("MILP-5")
    sdst = Data['sdsts']

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
                        X[j, l, h, z] = model.addVar(vtype=GRB.BINARY, name=f"X_{j}_{l}_{h}_{z}")

    # Objective Function: Minimize Cmax
    cmax = model.addVar(vtype=GRB.CONTINUOUS, name="Cmax")
    model.setObjective(cmax, GRB.MINIMIZE)

    # Constraints (3): Each operation is assigned to one and only one eligible machine
    for j in jobs:
        for l in operations_per_job[j]:
            model.addConstr(quicksum(Y[j, l, i] for i in machine_allocations[j, l]) == 1)

    # Constraints (4): Precedence relations between consecutive operations of the same job
    for j in jobs:
        for l in operations_per_job[j][1:]:
            model.addConstr(S[j, l] >= S[j, l-1] + quicksum(operations_times[j, l-1, i] * Y[j, l-1, i] for i in machine_allocations[j, l-1]))
    #
    # for j in jobs:
    #     for l in operations_per_job[j][1:]:
    #         common_machines = set(machine_allocations[j, l]) & set(machine_allocations[j, l-1])
    #         for i in common_machines:
    #             model.addConstr(
    #                 S[j, l] >= S[j, l - 1] + operations_times[j, l - 1, i] * Y[j, l - 1, i] + sdst[j, l - 1, j, l, i] *
    #                 Y[j, l - 1, i] * Y[j, l, i])


    # Constraints (5) & (6): No overlapping of operations on the same machine k
    for j in jobs:
        for l in operations_per_job[j]:
            for h in jobs:
                for z in operations_per_job[h]:
                    if not (j == h and l == z):
                        common_machines = set(machine_allocations[j, l]) & set(machine_allocations[h, z])
                        for i in common_machines:
                            model.addConstr(
                                S[j, l] >= S[h, z] + operations_times[h, z, i] + sdst[h, z, j, l, i] - ((
                                        2 - Y[j, l, i] - Y[h, z, i] + X[j, l, h, z]) * largeM))

    for j in jobs:
        for l in operations_per_job[j]:
            for h in jobs:
                for z in operations_per_job[h]:
                    if not (j == h and l == z):
                        common_machines = set(machine_allocations[j, l]) & set(machine_allocations[h, z])
                        for i in common_machines:
                            model.addConstr(
                                S[h, z] >= S[j, l] + operations_times[j, l, i] + sdst[j, l, h, z, i] - ((
                                        3 - Y[j, l, i] - Y[h, z, i] - X[j, l, h, z]) * largeM))


    # Constraints (7): Determine makespan
    for j in jobs:
        last_op = max(operations_per_job[j])
        model.addConstr(
            cmax >= S[j, last_op] + quicksum(
                operations_times[j, last_op, i] * Y[j, last_op, i] for i in machine_allocations[j, last_op]))

    model.params.TimeLimit = time_limit

    return model
