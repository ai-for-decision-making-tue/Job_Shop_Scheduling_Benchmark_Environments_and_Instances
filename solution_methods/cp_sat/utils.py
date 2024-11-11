import os
import sys

# Add the current file's directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import os
import datetime
import json

from ortools.sat.python import cp_model

DEFAULT_RESULTS_ROOT = os.getcwd() + "/results/cp_sat/"


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solution_methods."""

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


def solve_model(
    model: cp_model.CpModel, time_limit: float | int
) -> tuple[cp_model.CpSolver, int, int]:
    """
    Solves the given constraint programming model within the specified time limit.

    Args:
        model: The constraint programming model to solve.
        time_limit: The maximum time limit in seconds for solving the model.

    Returns:
        A tuple containing the solver object, the status of the solver, and the number of solution_methods found.
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solution_printer = SolutionPrinter()
    status = solver.Solve(model, solution_printer)
    solution_count = solution_printer.solution_count()
    return solver, status, solution_count


def output_dir_exp_name(parameters):
    if 'experiment_name' in parameters['output'] is not None:
        exp_name = parameters['output']['experiment_name']
    else:
        instance_name = parameters['instance']['problem_instance'].split('/')[-1].split('.')[0]
        time_limit = parameters['solver'].get('time_limit', 'default')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{instance_name}_time{time_limit}_{timestamp}"

    if 'folder_name' in parameters['output'] is not None:
        output_dir = parameters['output']['folder_name']
    else:
        output_dir = DEFAULT_RESULTS_ROOT
    return output_dir, exp_name


def results_saving(results, path):
    """
    Save the CP optimization results to a JSON file.

    Args:
        results: The results data to save.
        parameters: The configuration parameters dict.
    """

    # Generate a default experiment name based on instance and solve time if not provided
    os.makedirs(path, exist_ok=True)

    # Save results to JSON
    file_path = os.path.join(path, "milp_results.json")
    with open(file_path, "w") as outfile:
        json.dump(results, outfile, indent=4)