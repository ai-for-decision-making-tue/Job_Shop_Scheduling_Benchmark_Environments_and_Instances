"""
This file contains the OR-Tools model for the Flexible Job Shop Problem (FJSP).
This code has been adapted from the OR-Tools example for the FJSP, which can be found at:
https://github.com/google/or-tools/blob/stable/examples/python/flexible_job_shop_sat.py
"""

import collections

from ortools.sat.python import cp_model


def parse_file_jsp(filename: str) -> dict:
    """
    Parses a file containing job shop scheduling data and returns a dictionary
    with the number of jobs, number of machines, and the job operations.

    Args:
        filename (str): The name of the file to parse.

    Returns:
        dict: A dictionary containing the parsed data with the following keys:
            - "num_jobs": The number of jobs.
            - "num_machines": The number of machines.
            - "jobs": A list of job operations, where each job operation is a list
              with a single tuple representing the machine and processing time.

    """
    with open("./data/" + filename, "r") as f:
        num_jobs, num_machines = tuple(map(int, f.readline().strip().split()))
        jobs = []
        for _ in range(num_jobs):
            job_operations = []
            data_line = list(map(int, f.readline().split()))
            job_operations = [
                [(data_line[i + 1], data_line[i])] for i in range(0, len(data_line), 2)
            ]
            jobs.append(job_operations)

    return {"num_jobs": num_jobs, "num_machines": num_machines, "jobs": jobs}


def parse_file_fjsp(filename: str) -> dict:
    """
    Parses a file containing flexible job shop scheduling data and returns a
    dictionary with the number of jobs, number of machines, and the job operations.

    Args:
        filename (str): The name of the file to parse.

    Returns:
        dict: A dictionary containing the following keys:
            - "num_jobs" (int): The number of jobs.
            - "num_machines" (int): The number of machines.
            - "jobs" (list): A list of job operations. Each job operation is a list
              of tuples, where each tuple contains the processing time and machine
              index for an operation.
    """
    with open("./data/" + filename, "r") as f:
        num_jobs, num_machines = tuple(map(int, f.readline().strip().split()[:2]))
        jobs = []
        for _ in range(num_jobs):
            job_operations = []
            operation_data = list(map(int, f.readline().split()))
            index = 1
            while index < len(operation_data):
                # Extract machine and processing time data for each operation
                machines_for_task = operation_data[index]
                # below x - 1 to go from 1 as lowest index to 0 as lowest index
                job_machines = list(
                    map(
                        lambda x: x - 1,
                        operation_data[
                            index + 1 : index + 1 + machines_for_task * 2 : 2
                        ],
                    )
                )
                job_processingtimes = operation_data[
                    index + 2 : index + 2 + machines_for_task * 2 : 2
                ]
                operation_info = list(zip(job_processingtimes, job_machines))
                index += machines_for_task * 2 + 1
                job_operations.append(operation_info)
            jobs.append(job_operations)
    return {"num_jobs": num_jobs, "num_machines": num_machines, "jobs": jobs}


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        """Called at each new solution."""
        print(
            "Solution %i, time = %f s, objective = %i"
            % (self.__solution_count, self.WallTime(), self.ObjectiveValue())
        )
        self.__solution_count += 1

    def solution_count(self):
        return self.__solution_count


def fjsp_or_tools_model(data: dict) -> tuple[cp_model.CpModel, dict]:
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
    num_jobs = data["num_jobs"]
    num_machines = data["num_machines"]
    jobs = data["jobs"]
    all_jobs = range(num_jobs)
    all_machines = range(num_machines)

    model = cp_model.CpModel()

    horizon = 0
    for job in jobs:
        for task in job:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[0])
            horizon += max_task_duration

    print(f"Horizon = {horizon}")

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

            min_duration = task[0][0]
            max_duration = task[0][0]

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][0]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            # Create main interval for the task.
            suffix_name = "_j%i_t%i" % (job_id, task_id)
            start = model.NewIntVar(0, horizon, "start" + suffix_name)
            duration = model.NewIntVar(
                min_duration, max_duration, "duration" + suffix_name
            )
            end = model.NewIntVar(0, horizon, "end" + suffix_name)
            interval = model.NewIntervalVar(
                start, duration, end, "interval" + suffix_name
            )

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
                    alt_suffix = "_j%i_t%i_a%i" % (job_id, task_id, alt_id)
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


def solve_model(
    model: cp_model.CpModel, time_limit: float | int
) -> tuple[cp_model.CpSolver, int, int]:
    """
    Solves the given constraint programming model within the specified time limit.

    Args:
        model: The constraint programming model to solve.
        time_limit: The maximum time limit in seconds for solving the model.

    Returns:
        A tuple containing the solver object, the status of the solver, and the number of solutions found.
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solution_printer = SolutionPrinter()
    status = solver.Solve(model, solution_printer)
    solution_count = solution_printer.solution_count()
    return solver, status, solution_count
