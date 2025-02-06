import torch

from solution_methods.dispatching_rules.src.helper_functions import check_precedence_relations
from solution_methods.FJSP_DRL.src.env_test import EnvState
from solution_methods.FJSP_DRL.src.load_data import load_feats_from_sim


def schedule_operations_drl(simulationEnv, hgnn_model):
    """
    Schedule operations using a Deep Reinforcement Learning (DRL) approach.

    Args:
        simulationEnv: The simulation environment object.
        dispatching_policy: The policy used for generating actions.

    Returns:
        None
    """
    if not any_available_actions(simulationEnv):
        return

    # Load raw features from the simulation environment
    raw_features = load_feats_from_sim(
        simulationEnv.jobShopEnv,
        simulationEnv.jobShopEnv.nr_of_machines,
        simulationEnv.jobShopEnv.nr_of_operations,
    )

    # Extract operation and machine features
    operation_features = _load_operation_features(simulationEnv, raw_features)
    machine_features = _load_machine_features(simulationEnv, raw_features)

    # Construct environment state
    state = EnvState(
        batch_idxes=torch.tensor([0]),
        feat_opes_batch=operation_features,
        feat_mas_batch=machine_features,
        proc_times_batch=raw_features[0].unsqueeze(0),
        ope_ma_adj_batch=raw_features[1].unsqueeze(0),
        ope_pre_adj_batch=raw_features[2].unsqueeze(0),
        ope_sub_adj_batch=raw_features[3].unsqueeze(0),
        mask_job_procing_batch=torch.tensor([
            [
                any(ope.scheduled_end_time > simulationEnv.simulator.now
                    for ope in job.scheduled_operations)
                for job in simulationEnv.jobShopEnv.jobs
            ]
        ]),
        mask_job_finish_batch=torch.tensor([
            [
                len(job.operations) == len(job.scheduled_operations)
                for job in simulationEnv.jobShopEnv.jobs
            ]
        ], dtype=torch.bool),
        mask_ma_procing_batch=torch.tensor([
            [
                mach.next_available_time > simulationEnv.simulator.now
                for mach in simulationEnv.jobShopEnv.machines
            ]
        ], dtype=torch.bool),
        opes_appertain_batch=raw_features[4].unsqueeze(0),
        ope_step_batch=torch.tensor([
            [
                min((ope.operation_id for ope in job.operations
                     if ope not in job.scheduled_operations),
                    default=job.nr_of_ops)
                for job in simulationEnv.jobShopEnv.jobs
            ]
        ]),
        end_ope_biases_batch=torch.tensor([
            [job.operations[-1].operation_id for job in simulationEnv.jobShopEnv.jobs]
        ]),
        time_batch=simulationEnv.simulator.now,
        nums_opes_batch=torch.tensor([simulationEnv.jobShopEnv.nr_of_operations]),
    )

    # Get action from dispatcher
    action = hgnn_model.act(state, [], False, flag_train=False, flag_sample=False)
    operation_id, machine_id = action[0, :].item(), action[1, :].item()

    # Retrieve corresponding operation and machine
    operation = next((ope for ope in simulationEnv.jobShopEnv.operations
                      if ope.operation_id == operation_id), None)
    machine = next((mach for mach in simulationEnv.jobShopEnv.machines
                    if mach.machine_id == machine_id), None)

    if operation and machine:
        # Schedule the operation
        simulationEnv.jobShopEnv._scheduled_operations.append(operation)
        simulationEnv.simulator.process(simulationEnv.perform_operation(operation, machine))


def any_available_actions(simulationEnv):
    """
    Checks if there are any available actions in the given simulation environment.
    Args:
        simulationEnv: The simulation environment object.
    Returns:
        bool: True if there are available actions, False otherwise.
    """
    machines_available = [
        machine
        for machine in simulationEnv.jobShopEnv.machines
        if simulationEnv.machine_resources[machine.machine_id].count == 0
    ]
    if machines_available == []:
        return False
    for machine in machines_available:
        for job in simulationEnv.jobShopEnv.jobs:
            for operation in job.operations:
                if (
                    operation not in simulationEnv.processed_operations
                    and operation not in simulationEnv.jobShopEnv.scheduled_operations
                    and machine.machine_id in operation.processing_times
                ):
                    if check_precedence_relations(simulationEnv, operation):
                        return True
    return False


def _load_operation_features(simulationEnv, raw_features):
    """
    Load operation features for online dispatching.
    Args:
        simulationEnv: The simulation environment object.
        raw_features: The raw features for the operations.
    Returns:
        operation_features: The loaded operation features.
    """
    status = torch.zeros(simulationEnv.jobShopEnv.nr_of_operations)  # 1 status
    status[list(map(lambda x: x.operation_id, simulationEnv.jobShopEnv.scheduled_operations))] = 1.0
    status = status.unsqueeze(0)
    # 2 nr of neighboring machines
    op_mch_adj = raw_features[1].unsqueeze(0)
    nr_neighbor_machines = torch.count_nonzero(op_mch_adj, dim=2)
    # 3 processing time
    raw_proc_times = raw_features[0].unsqueeze(0)
    proc_times = torch.sum(raw_proc_times, dim=2).div(nr_neighbor_machines + 1e-9)
    # 4 num unscheduled operations
    num_unscheduled_per_job = list(
        map(
            lambda x: x.nr_of_ops - len(x.scheduled_operations),
            simulationEnv.jobShopEnv.jobs,
        )
    )
    unscheduled_ops = torch.tensor(
        [
            list(
                map(
                    lambda x: num_unscheduled_per_job[x.job_id],
                    simulationEnv.jobShopEnv.operations,
                )
            )
        ],
        dtype=torch.float32,
    )
    # 6 start time in partial schedule
    start_times = torch.bmm(
        proc_times.unsqueeze(1), raw_features[7].float().unsqueeze(0)
    ).squeeze()
    for each_ope in simulationEnv.jobShopEnv.operations:
        if each_ope.scheduling_information.__len__() == 0:
            if each_ope.predecessors.__len__() == 0:
                est_start_time = start_times[each_ope.operation_id]
            else:
                if each_ope.predecessors[0].scheduling_information.__len__() == 0:
                    est_start_time = (
                        start_times[each_ope.predecessors[0].operation_id]
                        + proc_times[0, each_ope.predecessors[0].operation_id]
                    )
                else:
                    est_start_time = (
                        each_ope.predecessors[0].scheduling_information["start_time"]
                        + each_ope.predecessors[0].scheduling_information[
                            "processing_time"
                        ]
                    )
        else:
            est_start_time = each_ope.scheduling_information["start_time"]
        start_times[each_ope.operation_id] = est_start_time
    start_times = start_times.unsqueeze(0)
    # 5 job completion time in partial schedule
    last_op_indices = [
        job.operations[-1].operation_id for job in simulationEnv.jobShopEnv.jobs
    ]
    last_op_batch = [
        last_op_indices[operation.job_id]
        for operation in simulationEnv.jobShopEnv.operations
    ]
    end_times = (start_times + proc_times)[:, last_op_batch]

    operation_features = torch.stack(
        (
            status,
            nr_neighbor_machines,
            proc_times,
            unscheduled_ops,
            end_times,
            start_times,
        ),
        dim=1,
    )
    return operation_features


def _load_machine_features(simulationEnv, raw_features):
    """
    Load machine features based on the given simulation environment and raw features.
    Args:
        simulationEnv (SimulationEnvironment): The simulation environment object.
        raw_features (list): The raw features for machine operations.
    Returns:
        torch.Tensor: The machine features tensor.
    """
    op_mch_adj = raw_features[1].unsqueeze(0)
    # 1 Number of neighboring operations
    nr_neighbor_operations = torch.count_nonzero(op_mch_adj, dim=1)
    # 2 next available time and # 3 utilization
    next_available_time = torch.zeros((1, simulationEnv.jobShopEnv.nr_of_machines))
    utilization = torch.zeros((1, simulationEnv.jobShopEnv.nr_of_machines))
    for each_mach in simulationEnv.jobShopEnv.machines:
        workload = sum(
            [
                ope_on_mach.scheduled_duration
                for ope_on_mach in each_mach.scheduled_operations
            ]
        )
        cur_time = simulationEnv.simulator.now
        workload = min(cur_time, workload)
        next_available_time[:, each_mach.machine_id] = each_mach.next_available_time
        utilization[:, each_mach.machine_id] = workload / (cur_time + 1e-9)

    machine_features = torch.stack(
        (nr_neighbor_operations, next_available_time, utilization), dim=1
    )
    return machine_features


def run_online_dispatcher(simulationEnv, hgnn_model):
    """Schedule simulator and schedule operations with the dispatching rules"""
    simulationEnv.simulator.process(simulationEnv.generate_online_job_arrivals())

    while True:
        schedule_operations_drl(simulationEnv, hgnn_model)
        yield simulationEnv.simulator.timeout(1)