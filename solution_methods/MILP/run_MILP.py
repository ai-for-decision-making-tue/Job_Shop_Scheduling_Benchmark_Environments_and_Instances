import argparse
import logging
import os

from visualization import gantt_chart, precedence_chart
from solution_methods.helper_functions import load_parameters, load_job_shop_env
from solution_methods.MILP.models import JSPmodel, FJSPmodel, FJSPSDSTmodel, FAJSPmodel
from solution_methods.MILP.utils import retrieve_decision_variables, results_saving, output_dir_exp_name

PARAM_FILE = os.path.abspath("../../configs/milp.toml")
logging.basicConfig(level=logging.INFO)

MODEL_MAP = {
    'fjsp_sdst': (FJSPSDSTmodel, "fjsp_sdst_milp"),
    'fjsp': (FJSPmodel, "fjsp_milp"),
    'fajsp': (FAJSPmodel, "fajsp_milp"),
    'jsp': (JSPmodel, "jsp_milp"),
    'fsp': (JSPmodel, "jsp_milp")  # FSP is solved as a special case of JSP
}



def run_MILP(jobShopEnv, **kwargs):
    """
    Solve the scheduling problem using a MILP model based on the problem instance provided.

    Args:
        jobShopEnv: The job shop environment to be optimized.
        kwargs: Additional keyword arguments including instance type and solver settings.

    Returns:
        tuple: Contains optimization results and the updated job shop environment.
    """
    try:
        instance_type = next((key for key in MODEL_MAP if key in kwargs['instance']['problem_instance']), None)
        if instance_type:
            model_class, method_name = MODEL_MAP[instance_type]  # Unpack tuple
            jobShopEnv = load_job_shop_env(kwargs['instance']['problem_instance'])
            model = getattr(model_class, method_name)(jobShopEnv, kwargs['solver']['time_limit'])
        else:
            raise ValueError("Unsupported problem instance type.")
    except Exception as e:
        logging.error(f"Error setting up MILP model: {e}")
        return None, jobShopEnv

    # Optimize the MILP model
    model.optimize()

    # Retrieve and update the decision variables in jobShopEnv
    results = retrieve_decision_variables(model, kwargs['solver']['time_limit'])
    jobShopEnv = model_class.update_env(jobShopEnv, results)

    return results, jobShopEnv


def main(param_file: str = PARAM_FILE):
    """
    Load parameters, run MILP optimization, and optionally plot and save results.

    Args:
        param_file (str): Path to the parameter configuration file.
    """
    try:
        parameters = load_parameters(param_file)
        logging.info(f"Parameters loaded from {param_file}.")
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    jobShopEnv = load_job_shop_env(parameters['instance'].get('problem_instance'))
    results, jobShopEnv = run_MILP(jobShopEnv, **parameters)

    if results:
        # Check output configuration and prepare output paths if needed
        output_config = parameters['output']
        save_gantt = output_config.get('save_gantt')
        save_results = output_config.get('save_results')
        show_gantt = output_config.get('show_gantt')
        show_precedences = output_config.get('show_precedences')

        if save_gantt or save_results:
            output_dir, exp_name = output_dir_exp_name(parameters)
            output_dir = os.path.join(output_dir, f"{exp_name}")
            os.makedirs(output_dir, exist_ok=True)

        # Draw precedence relations if required
        if show_precedences:
            precedence_chart.plot(jobShopEnv)

        # Plot Gantt chart if required
        if show_gantt or save_gantt:
            logging.info("Generating Gantt chart.")
            plt = gantt_chart.plot(jobShopEnv)

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
    parser = argparse.ArgumentParser(description="Run MILP")
    parser.add_argument(
        "-f", "--config_file",
        type=str,
        default=PARAM_FILE,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    main(param_file=args.config_file)
