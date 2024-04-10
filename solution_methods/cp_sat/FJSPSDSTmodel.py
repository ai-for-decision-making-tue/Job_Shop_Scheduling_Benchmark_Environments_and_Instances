"""
This file contains the OR-Tools model for the Flexible Job Shop Problem with Sequence Dependent Setup TImes (FJSP-SDST)
This code build upon the FJSPmodel, including the SDST constraints:
"""


import collections

from ortools.sat.python import cp_model


def update_env(jobShopEnv, vars, solver, status, solution_count, time_limit):
    """Update the job shop scheduling environment with the solution found by the solver."""

    # Map job operations to their processing times and machines (according to used OR-tools format)
    jobs_operations = [[[(value, key) for key, value in operation.processing_times.items()] for operation in job.operations] for job in jobShopEnv.jobs]
    # Create unique identifier for each operation (to deal with OR-tools format)
    operation_identifier = {(job_id, op_id): job_id * len(jobShopEnv.jobs[0].operations) + op_id for job_id, job in enumerate(jobShopEnv.jobs) for op_id, op in enumerate(job.operations)}
    starts, presences = vars["starts"], vars["presences"]

    # Check if a solution has been found
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("Solution:")

        schedule = []
        machine_schedule = {machine: {} for machine in jobShopEnv.machines}
        # Iterate through all jobs and tasks to construct the schedule
        for job_id in range(jobShopEnv.nr_of_jobs):
            job_schedule = {"job": job_id, "tasks": []}

            for task_id, task in enumerate(jobs_operations[job_id]):
                start_time = solver.Value(starts[(job_id, task_id)])

                for alt_id, (processing_time, machine_id) in enumerate(task):
                    if solver.Value(presences[(job_id, task_id, alt_id)]):
                        machine = jobShopEnv.get_machine(machine_id)
                        operation = jobShopEnv.get_operation(operation_identifier[(job_id, task_id)])
                        machine_schedule[machine][operation] = start_time
                        task_info = {"task": task_id, "start": start_time, "machine": machine_id,
                                     "duration": processing_time}
                        job_schedule["tasks"].append(task_info)
                        break  # Exit loop after finding the assigned machine

            schedule.append(job_schedule)

        # Update the environment with the task's scheduling information
        for machine, operations in machine_schedule.items():
            sorted_operations = sorted(operations.items(), key=lambda x: x[1])
            for idx, (operation, start_time) in enumerate(sorted_operations):
                processing_time = operation.processing_times[machine.machine_id]
                setup_time = 0 if idx == 0 else jobShopEnv._sequence_dependent_setup_times[machine.machine_id][
                    sorted_operations[idx - 1][0].operation_id][operation.operation_id]
                machine.add_operation_to_schedule_at_time(operation, start_time, processing_time, setup_time)

        # Compile results
        results = {
            "time_limit": str(time_limit),
            "status": status,
            "statusString": solver.StatusName(status),
            "objValue": solver.ObjectiveValue(),
            "runtime": solver.WallTime(),
            "numBranches": solver.NumBranches(),
            "conflicts": solver.NumConflicts(),
            "solution_methods": solution_count,
            "Schedule": schedule,
        }
        print(f"Optimal Schedule Length: {solver.ObjectiveValue()}")
    else:
        print("No solution found.")

    return jobShopEnv, results


def fjsp_sdst_cp_sat_model(jobShopEnv) -> tuple[cp_model.CpModel, dict]:
    """
    Creates a flexible job shop scheduling with sequence dependent setup times model using the OR-Tools library.
    """

    # Map job operations to their processing times and machines (according to used OR-tools format)
    jobs_operations = [[[(value, key) for key, value in operation.processing_times.items()] for operation in job.operations] for job in jobShopEnv.jobs]
    # Create unique identifier for each operation (to deal with OR-tools format)
    operation_identifier = {(job_id, op_id): job_id * len(jobShopEnv.jobs[0].operations) + op_id for job_id, job in enumerate(jobShopEnv.jobs) for op_id, op in enumerate(job.operations)}
    setup_times = jobShopEnv._sequence_dependent_setup_times

    # Computes horizon dynamically as the sum of all durations
    horizon = sum(max(alternative[0] for alternative in task) for job in jobs_operations for task in job)
    print(f"Horizon = {horizon}")

    # Create the model
    model = cp_model.CpModel()

    # Global storage of variables
    intervals_per_resources = collections.defaultdict(list)
    starts = {}  # indexed by (job_id, task_id).
    presences = {}  # indexed by (job_id, task_id, alt_id).
    job_ends = []
    machine_to_operations = collections.defaultdict(list)  # To keep track of transitions for setup time

    # Scan the jobs and create the relevant variables and intervals
    for job_id in range(jobShopEnv.nr_of_jobs):
        job = jobs_operations[job_id]
        num_tasks = len(job)
        previous_end = None
        for task_id in range(num_tasks):
            task = job[task_id]

            min_duration = task[0][0]
            max_duration = task[0][0]

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][0]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            # Create main interval for the task
            suffix_name = "_j%i_t%i" % (job_id, task_id)
            start = model.NewIntVar(0, horizon, "start" + suffix_name)
            duration = model.NewIntVar(min_duration, max_duration, "duration" + suffix_name)
            end = model.NewIntVar(0, horizon, "end" + suffix_name)
            interval = model.NewIntervalVar(start, duration, end, "interval" + suffix_name)

            # Store the start for the solution
            starts[(job_id, task_id)] = start

            # Add precedence with previous task in the same job
            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end

            # Create alternative intervals
            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = "_j%i_t%i_a%i" % (job_id, task_id, alt_id)
                    l_presence = model.NewBoolVar("presence" + alt_suffix)
                    l_start = model.NewIntVar(0, horizon, "start" + alt_suffix)
                    l_duration = task[alt_id][0]
                    l_end = model.NewIntVar(0, horizon, "end" + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_start, l_duration, l_end, l_presence, "interval" + alt_suffix
                    )
                    machine_to_operations[task[alt_id][1]].append((l_interval, job_id, task_id, alt_id, l_presence))
                    l_presences.append(l_presence)

                    # Link the primary/global variables with the local ones
                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)

                    # Add the local interval to the right machine
                    intervals_per_resources[task[alt_id][1]].append(l_interval)

                    # Store the presences for the solution
                    presences[(job_id, task_id, alt_id)] = l_presence

                # Select exactly one presence variable
                model.AddExactlyOne(l_presences)
            else:
                alt_id = 0
                presences[(job_id, task_id, 0)] = model.NewConstant(1)
                intervals_per_resources[task[0][1]].append(interval)
                pres = model.NewBoolVar(f"pres_j{job_id}_t{task_id}_a{alt_id}")
                machine_to_operations[task[alt_id][1]].append((interval, job_id, task_id, alt_id, pres))

        job_ends.append(previous_end)

    # Machine constraints and sequence-dependent setup times
    for machine_id, operations in machine_to_operations.items():
        intervals = [op[0] for op in operations]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

        for i, op_i in enumerate(operations):
            for j, op_j in enumerate(operations):
                if i >= j: continue  # Prevent duplicate constraints and self-comparison
                before_var = model.NewBoolVar(f"before_j{op_i[1]}_t{op_i[2]}_j{op_j[1]}_t{op_j[2]}_on_m{machine_id}")
                setup_time_ij = setup_times[machine_id][operation_identifier[(op_i[1], op_i[2])]][operation_identifier[(op_j[1], op_j[2])]]
                setup_time_ji = setup_times[machine_id][operation_identifier[(op_j[1], op_j[2])]][operation_identifier[(op_i[1], op_i[2])]]

                model.Add(op_i[0].EndExpr() + setup_time_ij <= op_j[0].StartExpr()).OnlyEnforceIf(before_var)
                model.Add(op_j[0].EndExpr() + setup_time_ji <= op_i[0].StartExpr()).OnlyEnforceIf(before_var.Not())

    # Makespan objective
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    return model, {"starts": starts, "presences": presences}