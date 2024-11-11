import argparse
import logging
import os

from plotting.drawer import plot_gantt_chart
from solution_methods.helper_functions import load_parameters, load_job_shop_env
from solution_methods.CP_SAT.utils import results_saving, output_dir_exp_name
from solution_methods.CP_SAT.models import FJSPSDSTmodel, FJSPmodel, JSPmodel
from solution_methods.CP_SAT.utils import solve_model

PARAM_FILE = os.path.abspath("../../configs/cp_sat.toml")
logging.basicConfig(level=logging.INFO)


def run_CP_SAT(jobShopEnv, **kwargs):
    """
    Solve the scheduling problem for the provided input file.
    """
    if kwargs["solver"]["model"] == "fjsp_sdst":
        model, vars = FJSPSDSTmodel.fjsp_sdst_cp_sat_model(jobShopEnv)
    elif kwargs["solver"]["model"] == "fjsp":
        model, vars = FJSPmodel.fjsp_cp_sat_model(jobShopEnv)
    elif kwargs["solver"]["model"] == "jsp" or kwargs["solver"]["model"] == "fsp":
        model, vars = JSPmodel.jsp_cp_sat_model(jobShopEnv)
    else:
        raise ValueError(f"Unknown model type: {kwargs['algorithm']['model']}")

    solver, status, solution_count = solve_model(model, kwargs["solver"]["time_limit"])

    # Update jobShopEnv with found solution
    if kwargs["solver"]["model"] == "fjsp_sdst":
        jobShopEnv, results = FJSPSDSTmodel.update_env(jobShopEnv, vars, solver, status, solution_count, kwargs["solver"]["time_limit"])
    elif kwargs["solver"]["model"] == "fjsp":
        jobShopEnv, results = FJSPmodel.update_env(jobShopEnv, vars, solver, status, solution_count, kwargs["solver"]["time_limit"])
    elif kwargs["solver"]["model"] == "jsp" or kwargs["solver"]["model"] == "fsp":
        jobShopEnv, results = JSPmodel.update_env(jobShopEnv, vars, solver, status, solution_count, kwargs["solver"]["time_limit"])

    return results, jobShopEnv


def main(param_file=PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
        logging.info(f"Parameters loaded from {param_file}.")
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    jobShopEnv = load_job_shop_env(parameters['instance'].get('problem_instance'))
    results, jobShopEnv = run_CP_SAT(jobShopEnv, **parameters)

    if results:
        # Check output configuration and prepare output paths if needed
        output_config = parameters['output']
        save_gantt = output_config.get('save_gantt')
        save_results = output_config.get('save_results')
        show_gantt = output_config.get('show_gantt')

        if save_gantt or save_results:
            output_dir, exp_name = output_dir_exp_name(parameters)
            output_dir = os.path.join(output_dir, f"{exp_name}")
            os.makedirs(output_dir, exist_ok=True)

        # Plot Gantt chart if required
        if show_gantt or save_gantt:
            logging.info("Generating Gantt chart.")
            plt = plot_gantt_chart(jobShopEnv)

            if save_gantt:
                plt.savefig(output_dir + "/gantt.png")
                logging.info(f"Gantt chart saved to {output_dir}")

            if show_gantt:
                plt.show()

        # Save results if enabled
        if save_results:
            results_saving(results, output_dir)
            logging.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OR-Tools CP-SAT")
    parser.add_argument(
        "-f", "--config_file",
        type=str,
        default=PARAM_FILE,
        help="Path to the configuration file.",
        )
    args = parser.parse_args()
    main(param_file=args.config_file)
