"""
This file contains the OR-Tools model for the Job Shop Problem (JSP).
This code has been adapted from the OR-Tools example for the JSP, which can be found at:
https://developers.google.com/optimization/scheduling/job_shop
"""

import collections

from ortools.sat.python import cp_model


def update_env(jobShopEnv, vars, solver, status, solution_count, time_limit):
    # Gather Final Schedule

    # Map job operations to their processing times and machines (according to used OR-tools format)
    jobs_operations = [[(k, v) for operation in job.operations for k, v in operation.processing_times.items()] for job in jobShopEnv.jobs]
    all_tasks = vars['all_tasks']

    # Check if a solution has been found
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("Solution:")

        schedule = []
        # Iterate through all jobs and tasks to construct the schedule
        for job_id, job_operations in enumerate(jobs_operations):
            job_info = {"job": job_id, "tasks": []}

            for task_id, task in enumerate(job_operations):
                start_time = solver.Value(all_tasks[job_id, task_id].start)
                machine_id, processing_time = task[0], task[1]

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
                setup_time = 0  # No setup time required for JSP/FSP
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
            "Schedule": schedule
        }

        # Finally print the solution found.
        print(f"Optimal Schedule Length: {solver.ObjectiveValue}")
    else:
        print("No solution found.")

    return jobShopEnv, results


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