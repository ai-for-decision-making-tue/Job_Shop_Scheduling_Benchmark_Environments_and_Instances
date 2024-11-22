"""
This file contains the OR-Tools model for the Flexible Job Shop Problem (FJSP).
This code has been adapted from the OR-Tools example for the FJSP, which can be found at:
https://github.com/google/or-tools/blob/stable/examples/python/flexible_job_shop_sat.py
"""

import collections

from ortools.sat.python import cp_model


def update_env(jobShopEnv, vars, solver, status, solution_count, time_limit):
    """Update the job shop scheduling environment with the solution found by the solver."""

    # Map job operations to their processing times and machines (according to used OR-tools format)
    jobs_operations = [[[(value, key) for key, value in operation.processing_times.items()] for operation in job.operations] for job in jobShopEnv.jobs]
    starts, presences = vars["starts"], vars["presences"]

    # Check if a solution has been found
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("Solution:")

        schedule = []
        # Iterate through all jobs and tasks to construct the schedule
        for job_id, job_operations in enumerate(jobs_operations):
            job_info = {"job": job_id, "tasks": []}

            for task_id, alternatives in enumerate(job_operations):
                start_time = solver.Value(starts[(job_id, task_id)])
                machine_id, processing_time = -1, -1  # Initialize as not found

                # Identify the chosen machine and processing time for the task
                for alt_id, (alt_time, alt_machine_id) in enumerate(alternatives):
                    if solver.Value(presences[(job_id, task_id, alt_id)]):
                        processing_time, machine_id = alt_time, alt_machine_id
                        break  # Exit the loop once the selected alternative is found

                # Append task information to the job schedule
                task_info = {
                    "task": task_id,
                    "start": start_time,
                    "machine": machine_id,
                    "duration": processing_time,
                }
                job_info["tasks"].append(task_info)

                # Update the environment with the task's scheduling information
                job = jobShopEnv.get_job(job_id)
                machine = jobShopEnv.get_machine(machine_id)
                operation = job.operations[task_id]
                setup_time = 0  # No setup time required for FJSP
                machine.add_operation_to_schedule_at_time(operation, start_time, processing_time, setup_time)

            schedule.append(job_info)

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


def fjsp_cp_sat_model(jobShopEnv) -> tuple[cp_model.CpModel, dict]:
    """
    Creates a flexible job shop scheduling model using the OR-Tools library.
    """

    # Map job operations to their processing times and machines (according to used OR-tools format)
    jobs_operations = [[[(value, key) for key, value in operation.processing_times.items()] for operation in job.operations] for job in jobShopEnv.jobs]

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
                intervals_per_resources[task[0][1]].append(interval)
                presences[(job_id, task_id, 0)] = model.NewConstant(1)

        job_ends.append(previous_end)

    # Create machines constraints
    for machine_id in range(jobShopEnv.nr_of_machines):
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Makespan objective
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)
    return model, {"starts": starts, "presences": presences}
