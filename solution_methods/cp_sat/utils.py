from ortools.sat.python import cp_model


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