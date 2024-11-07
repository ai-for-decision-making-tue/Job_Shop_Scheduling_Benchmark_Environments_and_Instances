import os
import datetime
import json

from gurobipy import GRB


DEFAULT_RESULTS_ROOT = os.getcwd() + "/results/milp"


def retrieve_decision_variables(model, time_limit):
    """retrieves the decision values from a solved MILP model"""

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
        'time_limit': str(time_limit),
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

    return results


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
    Save the MILP optimization results to a JSON file.

    Args:
        results: The results data to save.
        path: The path to save the results to.
    """

    # Generate a default experiment name based on instance and solve time if not provided
    os.makedirs(path, exist_ok=True)

    # Save results to JSON
    file_path = os.path.join(path, "cp_sat_results.json")
    with open(file_path, "w") as outfile:
        json.dump(results, outfile, indent=4)