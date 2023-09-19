import argparse
from gurobipy import GRB
from solutions.MILP import FJSPmodel
from solutions.helper_functions import load_parameters
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
DEFAULT_RESULTS_ROOT = "./results/milp"
PARAM_FILE = "configs/milp.toml"


def main(param_file=PARAM_FILE):
    """
    Solve the FJSP problem for the provided input file.

    Args:
        filename (str): Path to the file containing the FJSP data.

    Returns:
        None. Prints the optimization result.
    """
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    folder = DEFAULT_RESULTS_ROOT

    exp_name = "gurobi_" + str(parameters['solver']["time_limit"]) + "/" + \
                str(parameters['instance']['problem_instance'])

    data = FJSPmodel.parse_file(parameters['instance']['problem_instance'])
    model = FJSPmodel.fjsp_milp(data, parameters['solver']['time_limit'])
    model.optimize()

    # Status dictionary mapping
    status_dict = {
        GRB.OPTIMAL: 'OPTIMAL',
        GRB.INFEASIBLE: 'INFEASIBLE',
        GRB.INF_OR_UNBD: 'INF_OR_UNBD',
        GRB.UNBOUNDED: 'UNBOUNDED',
        GRB.CUTOFF: 'CUTOFF',
        GRB.ITERATION_LIMIT: 'ITERATION_LIMIT',
        GRB.NODE_LIMIT: 'NODE_LIMIT',
        GRB.TIME_LIMIT: 'TIME_LIMIT',
        GRB.SOLUTION_LIMIT: 'SOLUTION_LIMIT',
        GRB.INTERRUPTED: 'INTERRUPTED',
        GRB.NUMERIC: 'NUMERIC',
        GRB.SUBOPTIMAL: 'SUBOPTIMAL',
        GRB.INPROGRESS: 'INPROGRESS',
        GRB.USER_OBJ_LIMIT: 'USER_OBJ_LIMIT'
    }

    results = {
        'time_limit': str(parameters['solver']["time_limit"]),
        'status': model.status,
        'statusString': status_dict.get(model.status, 'UNKNOWN'),
        'objValue': model.objVal if model.status == GRB.OPTIMAL else None,
        'objBound': model.ObjBound if hasattr(model, "ObjBound") else None,
        'variables': {},
        'runtime': model.Runtime,
        'nodeCount': model.NodeCount,
        'iterationCount': model.IterCount,
    }

    for v in model.getVars():
        results['variables'][v.varName] = v.x

    # Ensure the directory exists; create if not
    dir_path = os.path.join(folder, exp_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Specify the full path for the file
    file_path = os.path.join(dir_path, "milp_results.json")

    # Save results to JSON (will create or overwrite the file)
    with open(file_path, "w") as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MILP")
    parser.add_argument("config_file",
                        metavar='-f',
                        type=str,
                        nargs="?",
                        default=PARAM_FILE,
                        help="path to config JSON",
                        )
    args = parser.parse_args()
    main(param_file=args.config_file)