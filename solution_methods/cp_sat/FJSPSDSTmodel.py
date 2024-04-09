"""
This file contains the OR-Tools model for the Flexible Job Shop Problem (FJSP).
This code has been adapted from the OR-Tools example for the FJSP, which can be found at:
https://github.com/google/or-tools/blob/stable/examples/python/flexible_job_shop_sat.py
"""

import collections
from ortools.sat.python import cp_model


def update_env(jobShopEnv, vars, solver, status, solution_count, time_limit):
    # Gather Final Schedule
    all_jobs = range(jobShopEnv.nr_of_jobs)
    jobs = [[[(value, key) for key, value in operation.processing_times.items()] for operation in job.operations] for
            job in jobShopEnv.jobs]
    starts = vars["starts"]
    presences = vars["presences"]

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution:")

        schedule = []
        for job_id in all_jobs:
            job_info = {"job": job_id, "tasks": []}
            for task_id in range(len(jobs[job_id])):
                start_time = solver.Value(starts[(job_id, task_id)])
                machine_id = -1
                processing_time = -1
                for alt_id in range(len(jobs[job_id][task_id])):
                    if solver.Value(presences[(job_id, task_id, alt_id)]):
                        processing_time = jobs[job_id][task_id][alt_id][0]
                        machine_id = jobs[job_id][task_id][alt_id][1]

                task_info = {
                    "task": task_id,
                    "start": start_time,
                    "machine": machine_id,
                    "duration": processing_time,
                }
                job_info["tasks"].append(task_info)

                # add schedule info to environment
                job = jobShopEnv.get_job(job_id)
                machine = jobShopEnv.get_machine(machine_id)
                operation = job.operations[task_id]
                setup_time = 0
                machine.add_operation_to_schedule_at_time(operation, start_time, processing_time, setup_time)
            schedule.append(job_info)

        # Status dictionary mapping
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

        # Finally print the solution found.
        print(f"Optimal Schedule Length: {solver.ObjectiveValue}")
    else:
        print("No solution found.")

    return jobShopEnv, results


def fjsp_sdst_cp_sat_model(jobShopEnv) -> tuple[cp_model.CpModel, dict]:
    """
    Creates a flexible job shop scheduling model using the OR-Tools library.

    Args:
        data (dict): A dictionary containing the input data for the flexible job shop scheduling problem.
            The dictionary should have the following keys:
            - "num_jobs" (int): The number of jobs in the problem.
            - "num_machines" (int): The number of machines in the problem.
            - "jobs" (list): A list of jobs, where each job is represented as a list of tasks.
                Each task is represented as a list of alternatives, where each alternative is a tuple
                containing the duration of the task and the machine on which it can be executed.

    Returns:
        tuple[cp_model.CpModel, dict]: A tuple containing the flexible job shop scheduling model and a dictionary
        with the variables and intervals created during the model construction. The dictionary has the
        following keys:
        - "starts" (dict): A dictionary mapping (job_id, task_id) tuples to the corresponding start variables.
        - "presences" (dict): A dictionary mapping (job_id, task_id, alt_id) tuples to the corresponding
            presence variables.

    """
    jobs = [[[(value, key) for key, value in operation.processing_times.items()] for operation in job.operations] for
            job in jobShopEnv.jobs]
    num_jobs = jobShopEnv.nr_of_jobs
    num_machines = jobShopEnv.nr_of_machines
    all_jobs = range(num_jobs)
    all_machines = range(num_machines)

    setup_times = jobShopEnv._sequence_dependent_setup_times

    # Computes horizon dynamically as the sum of all durations
    horizon = sum(max(alternative[0] for alternative in task) for job in jobs for task in job)
    print(f"Horizon = {horizon}")

    # Create the model
    model = cp_model.CpModel()

    # Global storage of variables.
    intervals_per_resources = collections.defaultdict(list)
    starts = {}  # indexed by (job_id, task_id).
    presences = {}  # indexed by (job_id, task_id, alt_id).
    job_ends = []

    # Scan the jobs and create the relevant variables and intervals.
    for job_id in all_jobs:
        job = jobs[job_id]
        num_tasks = len(job)
        previous_end = None
        for task_id in range(num_tasks):
            task = job[task_id]

            min_duration = min([duration for duration, id in task])
            max_duration = max([duration for duration, id in task])

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            # Create main interval for the task.
            suffix_name = "_job%i_task%i" % (job_id, task_id)
            start = model.NewIntVar(0, horizon, "start" + suffix_name)
            duration = model.NewIntVar(min_duration, max_duration, "duration" + suffix_name)
            end = model.NewIntVar(0, horizon, "end" + suffix_name)
            interval = model.NewIntervalVar(start, duration, end, "interval" + suffix_name)

            # Store the start for the solution.
            starts[(job_id, task_id)] = start

            # Add precedence with previous task in the same job.
            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end

            # Create alternative intervals.
            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = "_job%i_task%i_alt%i" % (job_id, task_id, alt_id)
                    l_presence = model.NewBoolVar("presence" + alt_suffix)
                    l_start = model.NewIntVar(0, horizon, "start" + alt_suffix)
                    l_duration = task[alt_id][0]
                    l_end = model.NewIntVar(0, horizon, "end" + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_start, l_duration, l_end, l_presence, "interval" + alt_suffix
                    )
                    l_presences.append(l_presence)

                    # Link the primary/global variables with the local ones.
                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)

                    # Add the local interval to the right machine.
                    intervals_per_resources[task[alt_id][1]].append(l_interval)

                    # Store the presences for the solution.
                    presences[(job_id, task_id, alt_id)] = l_presence

                # Select exactly one presence variable.
                model.AddExactlyOne(l_presences)
            else:
                intervals_per_resources[task[0][1]].append(interval)
                presences[(job_id, task_id, 0)] = model.NewConstant(1)

        job_ends.append(previous_end)

    print('hoi')

    # Create machines constraints.
    for machine_id in all_machines:
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Makespan objective
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)
    return model, {"starts": starts, "presences": presences}
