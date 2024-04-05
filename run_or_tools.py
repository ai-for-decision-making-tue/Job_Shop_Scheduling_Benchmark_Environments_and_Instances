import argparse
import json
import logging
import os

from solution_methods.helper_functions import load_parameters
from solution_methods.or_tools.FJSPmodel import fjsp_or_tools_model, parse_file_fjsp, parse_file_jsp, solve_model

logging.basicConfig(level=logging.INFO)
DEFAULT_RESULTS_ROOT = "./results/or_tools"
PARAM_FILE = "configs/or_tools.toml"


def main(param_file: str = PARAM_FILE) -> None:
    """
    Solve the (F)JSP problem for the provided input file.

    Args:
        filename (str): Path to the file containing the (F)JSP data.

    Returns:
        None. Prints the optimization result.
    """
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    folder = DEFAULT_RESULTS_ROOT

    exp_name = (
        "or_tools_" + str(parameters["solver"]["time_limit"]) + "/" + str(parameters["instance"]["problem_instance"])
    )

    if "fjsp" in str(parameters["instance"]["problem_instance"]):
        data = parse_file_fjsp(parameters["instance"]["problem_instance"])
        model, vars = fjsp_or_tools_model(data)
    elif any(
        scheduling_problem in str(parameters["instance"]["problem_instance"])
        for scheduling_problem in ["jsp", "fsp"]
    ):
        data = parse_file_jsp(parameters["instance"]["problem_instance"])
        model, vars = fjsp_or_tools_model(data)
    solver, status, solution_count = solve_model(
        model, parameters["solver"]["time_limit"]
    )

    # Gather Final Schedule
    all_jobs = range(data["num_jobs"])
    jobs = data["jobs"]
    starts = vars["starts"]
    presences = vars["presences"]

    schedule = []
    for job_id in all_jobs:
        job_info = {"job": job_id, "tasks": []}
        print("Job %i:" % job_id)
        for task_id in range(len(jobs[job_id])):
            start_value = solver.Value(starts[(job_id, task_id)])
            machine = -1
            duration = -1
            selected = -1
            for alt_id in range(len(jobs[job_id][task_id])):
                if solver.Value(presences[(job_id, task_id, alt_id)]):
                    duration = jobs[job_id][task_id][alt_id][0]
                    machine = jobs[job_id][task_id][alt_id][1]
                    selected = alt_id
            print(
                "  task_%i_%i starts at %i (alt %i, machine %i, duration %i)"
                % (job_id, task_id, start_value, selected, machine, duration)
            )
            task_info = {
                "task": task_id,
                "start": start_value,
                "machine": machine,
                "duration": duration,
            }
            job_info["tasks"].append(task_info)
        schedule.append(job_info)
    # Status dictionary mapping
    results = {
        "time_limit": str(parameters["solver"]["time_limit"]),
        "status": status,
        "statusString": solver.StatusName(status),
        "objValue": solver.ObjectiveValue(),
        "runtime": solver.WallTime(),
        "numBranches": solver.NumBranches(),
        "conflicts": solver.NumConflicts(),
        "solution_methods": solution_count,
        "Schedule": schedule,
    }

    # Ensure the directory exists; create if not
    dir_path = os.path.join(folder, exp_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Specify the full path for the file
    file_path = os.path.join(dir_path, "CP_results.json")

    # Save results to JSON (will create or overwrite the file)
    with open(file_path, "w") as outfile:
        json.dump(results, outfile, indent=4)

    # Print Results
    print("Solve status: %s" % solver.StatusName(status))
    print("Optimal objective value: %i" % solver.ObjectiveValue())
    print("Statistics")
    print("  - conflicts : %i" % solver.NumConflicts())
    print("  - branches  : %i" % solver.NumBranches())
    print("  - wall time : %f s" % solver.WallTime())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OR-Tools CP-SAT")
    parser.add_argument(
        "config_file",
        metavar="-f",
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )
    args = parser.parse_args()
    main(param_file=args.config_file)
