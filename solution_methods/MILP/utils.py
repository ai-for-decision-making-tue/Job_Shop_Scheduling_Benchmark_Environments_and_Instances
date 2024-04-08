from gurobipy import GRB


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
