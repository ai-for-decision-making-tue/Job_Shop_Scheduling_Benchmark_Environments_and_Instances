import argparse
import json
import logging
import os

from gurobipy import GRB

from solution_methods.helper_functions import load_parameters
from solution_methods.MILP import FJSPmodel, FJSPSDSTmodel, JSPmodel

logging.basicConfig(level=logging.INFO)
DEFAULT_RESULTS_ROOT = "./results/milp"
PARAM_FILE = "configs/milp.toml"


def run_method(folder, exp_name, **kwargs):
    """
    Solve the FJSP problem for the provided input file.

    Args:
        filename (str): Path to the file containing the FJSP data.

    Returns:
        None. Prints the optimization result.
    """

    if 'fjsp_sdst' in str(kwargs['instance']['problem_instance']):
        data = FJSPSDSTmodel.parse_file(kwargs['instance']['problem_instance'])
        model = FJSPSDSTmodel.fjsp_sdst_milp(data, kwargs['solver']['time_limit'])
    elif 'fjsp' in str(kwargs['instance']['problem_instance']):
        data = FJSPmodel.parse_file(kwargs['instance']['problem_instance'])
        model = FJSPmodel.fjsp_milp(data, kwargs['solver']['time_limit'])
    elif 'jsp' in str(kwargs['instance']['problem_instance']):
        data = JSPmodel.parse_file(kwargs['instance']['problem_instance'])
        model = JSPmodel.jsp_milp(data, kwargs['solver']['time_limit'])

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
        'time_limit': str(kwargs['solver']["time_limit"]),
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


def main(param_file=PARAM_FILE):

    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    folder = DEFAULT_RESULTS_ROOT

    exp_name = "gurobi_" + str(parameters['solver']["time_limit"]) + "/" + str(parameters['instance']['problem_instance'])

    run_method(folder, exp_name, **parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MILP")
    parser.add_argument("config_file",
                        metavar='-f',
                        type=str,
                        nargs="?",
                        default=PARAM_FILE,
                        help="path to config file",
                        )
    args = parser.parse_args()
    main(param_file=args.config_file)
