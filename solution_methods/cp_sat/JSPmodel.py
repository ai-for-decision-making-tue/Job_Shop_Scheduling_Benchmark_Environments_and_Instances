"""
This file contains the OR-Tools model for the Job Shop Problem (JSP).
This code has been adapted from the OR-Tools example for the JSP, which can be found at:
https://developers.google.com/optimization/scheduling/job_shop
"""

import collections

from ortools.sat.python import cp_model


def update_env(jobShopEnv, vars, solver, status, solution_count, time_limit):
    # Gather Final Schedule
    all_tasks = vars['all_tasks']
    jobs_data = [[(k, v) for operation in job.operations for k, v in operation.processing_times.items()] for job in jobShopEnv.jobs]
    num_machines = jobShopEnv.nr_of_machines
    all_machines = range(num_machines)

    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple(
        "assigned_task_type", "start job index duration"
    )

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution:")
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=solver.Value(all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1],
                    )
                )

        # Create per machine output lines.
        output = ""
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = "Machine " + str(machine) + ": "
            sol_line = "           "

            for assigned_task in assigned_jobs[machine]:
                name = f"job_{assigned_task.job}_task_{assigned_task.index}"
                # add spaces to output to align columns.
                sol_line_tasks += f"{name:15}"

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = f"[{start},{start + duration}]"
                # add spaces to output to align columns.
                sol_line += f"{sol_tmp:15}"

            sol_line += "\n"
            sol_line_tasks += "\n"
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print(f"Optimal Schedule Length: {solver.ObjectiveValue}")
        print(output)
    else:
        print("No solution found.")

    return jobShopEnv


def jsp_cp_sat_model(jobShopEnv) -> tuple[cp_model.CpModel, dict]:
    """
    Creates a job shop scheduling model using the OR-Tools library.
    """

    jobs_data = [[(k, v) for operation in job.operations for k, v in operation.processing_times.items()] for job in jobShopEnv.jobs]
    num_machines = jobShopEnv.nr_of_machines
    all_machines = range(num_machines)

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)
    print(f"Horizon = {horizon}")

    # Create the model
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple("task_type", "start end interval")

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine, duration = task
            suffix = f"_{job_id}_{task_id}"
            start_var = model.NewIntVar(0, horizon, "start" + suffix)
            end_var = model.NewIntVar(0, horizon, "end" + suffix)
            interval_var = model.NewIntervalVar(
                start_var, duration, end_var, "interval" + suffix
            )
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var
            )
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)],)
    model.Minimize(makespan)
    return model, {"all_tasks": all_tasks}