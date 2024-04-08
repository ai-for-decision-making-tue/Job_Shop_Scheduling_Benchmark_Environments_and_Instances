import argparse
import json
import logging
import os

from solution_methods.helper_functions import load_parameters, load_job_shop_env
from solution_methods.MILP import FJSPmodel, FJSPSDSTmodel, JSPmodel
from solution_methods.MILP.utils import retrieve_decision_variables
from plotting.drawer import draw_gantt_chart

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

    try:
        jobShopEnv = load_job_shop_env(kwargs['instance']['problem_instance'])
        if 'fjsp_sdst' in str(kwargs['instance']['problem_instance']):
            model = FJSPSDSTmodel.fjsp_sdst_milp(jobShopEnv, kwargs['solver']['time_limit'])
        elif 'fjsp' in str(kwargs['instance']['problem_instance']):
            model = FJSPmodel.fjsp_milp(jobShopEnv, kwargs['solver']['time_limit'])
        elif 'jsp' in str(kwargs['instance']['problem_instance']):
            model = JSPmodel.jsp_milp(jobShopEnv, kwargs['solver']['time_limit'])
    except Exception as e:
        print(f"MILP only implemented for 'jsp', 'fjs (as jsp)', 'fjsp', 'fjsp_sdst': {e}")

    model.optimize()

    # Retrieve the decisions made by the MILP model
    results = retrieve_decision_variables(model, kwargs['solver']['time_limit'])

    # Update jobShopEnv
    if 'fjsp_sdst' in str(kwargs['instance']['problem_instance']):
        jobShopEnv = FJSPSDSTmodel.update_env(jobShopEnv, results)
    elif 'fjsp' in str(kwargs['instance']['problem_instance']):
        jobShopEnv = FJSPmodel.update_env(jobShopEnv, results)
    elif 'jsp' in str(kwargs['instance']['problem_instance']):
        jobShopEnv = JSPmodel.update_env(jobShopEnv, results)

    # Plot ganttchart
    if kwargs['output']['plotting']:
        draw_gantt_chart(jobShopEnv)

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
