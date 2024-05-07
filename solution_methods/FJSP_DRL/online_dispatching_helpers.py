import argparse
import os

import torch

from solution_methods.dispatching_rules.helper_functions import \
    check_precedence_relations
from solution_methods.FJSP_DRL.env_test import EnvState
from solution_methods.FJSP_DRL.load_data import load_feats_from_sim
from solution_methods.FJSP_DRL.PPO import HGNNScheduler
from solution_methods.helper_functions import load_parameters


def load_model(device: str, model_parameters: dict, test_parameters: dict):
    """
    Loads a trained policy model for online dispatching.

    Args:
        device (str): The device to load the model on.
        model_parameters (dict): Parameters for the model.
        test_parameters (dict): Parameters for testing.

    Returns:
        HGNNScheduler: The loaded HGNNScheduler model.

    Raises:
        FileNotFoundError: If the trained policy file is not found.
    """
    # Configure default device
    device = torch.device(device)
    torch.set_default_tensor_type(
        "torch.cuda.FloatTensor" if device.type == "cuda" else "torch.FloatTensor"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Load trained policy
    trained_policy = os.getcwd() + test_parameters["trained_policy"]
    if trained_policy.endswith(".pt"):
        if device.type == "cuda":
            policy = torch.load(trained_policy)
        else:
            policy = torch.load(trained_policy, map_location="cpu")

        model_parameters["actor_in_dim"] = (
            model_parameters["out_size_ma"] * 2 + model_parameters["out_size_ope"] * 2
        )
        model_parameters["critic_in_dim"] = (
            model_parameters["out_size_ma"] + model_parameters["out_size_ope"]
        )

        hgnn_model = HGNNScheduler(model_parameters).to(device)
        print("\nloading saved model:", trained_policy)
        hgnn_model.load_state_dict(policy)

        return hgnn_model


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
        for machine in simulationEnv.JobShop.machines
        if simulationEnv.machine_resources[machine.machine_id].count == 0
    ]
    if machines_available == []:
        return False
    for machine in machines_available:
        for job in simulationEnv.JobShop.jobs:
            for operation in job.operations:
                if (
                    operation not in simulationEnv.processed_operations
                    and operation not in simulationEnv.JobShop.scheduled_operations
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
    status = torch.zeros(simulationEnv.JobShop.nr_of_operations)  # 1 status
    status[
        list(map(lambda x: x.operation_id, simulationEnv.JobShop.scheduled_operations))
    ] = 1.0
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
            simulationEnv.JobShop.jobs,
        )
    )
    unscheduled_ops = torch.tensor(
        [
            list(
                map(
                    lambda x: num_unscheduled_per_job[x.job_id],
                    simulationEnv.JobShop.operations,
                )
            )
        ],
        dtype=torch.float32,
    )
    # 6 start time in partial schedule
    start_times = torch.bmm(
        proc_times.unsqueeze(1), raw_features[7].float().unsqueeze(0)
    ).squeeze()
    for each_ope in simulationEnv.JobShop.operations:
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
        job.operations[-1].operation_id for job in simulationEnv.JobShop.jobs
    ]
    last_op_batch = [
        last_op_indices[operation.job_id]
        for operation in simulationEnv.JobShop.operations
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
    next_available_time = torch.zeros((1, simulationEnv.JobShop.nr_of_machines))
    utilization = torch.zeros((1, simulationEnv.JobShop.nr_of_machines))
    for each_mach in simulationEnv.JobShop.machines:
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


def schedule_operations_drl(simulationEnv, dispatcher):
    """
    Schedule operations using Deep Reinforcement Learning (DRL) approach.

    Args:
        simulationEnv: The simulation environment object.
        dispatcher: The dispatcher object responsible for generating actions.

    Returns:
        None
    """
    # Check if there are any available actions
    if not any_available_actions(simulationEnv):
        return
    raw_features = load_feats_from_sim(
        simulationEnv.JobShop,
        simulationEnv.JobShop.nr_of_machines,
        simulationEnv.JobShop.nr_of_operations,
    )
    #### Operation features
    operation_features = _load_operation_features(simulationEnv, raw_features)
    ##### Machine features
    machine_features = _load_machine_features(simulationEnv, raw_features)
    # Load state
    state = EnvState(
        batch_idxes=torch.tensor([0]),
        feat_opes_batch=operation_features,
        feat_mas_batch=machine_features,
        proc_times_batch=raw_features[0].unsqueeze(0),
        ope_ma_adj_batch=raw_features[1].unsqueeze(0),
        ope_pre_adj_batch=raw_features[2].unsqueeze(0),
        ope_sub_adj_batch=raw_features[3].unsqueeze(0),
        mask_job_procing_batch=torch.tensor(
            [
                [
                    (
                        True
                        if this_job.next_ope_earliest_begin_time
                        > simulationEnv.simulator.now
                        else False
                    )
                    for this_job in simulationEnv.JobShop.jobs
                ]
            ]
        ),
        mask_job_finish_batch=torch.tensor(
            [
                [
                    (
                        True
                        if this_job.operations.__len__()
                        == this_job.scheduled_operations.__len__()
                        else False
                    )
                    for this_job in simulationEnv.JobShop.jobs
                ]
            ],
            dtype=torch.bool,
        ),
        mask_ma_procing_batch=torch.tensor(
            [
                [
                    (
                        True
                        if this_mach.next_available_time > simulationEnv.simulator.now
                        else False
                    )
                    for this_mach in simulationEnv.JobShop.machines
                ]
            ],
            dtype=torch.bool,
        ),
        opes_appertain_batch=raw_features[4].unsqueeze(0),
        ope_step_batch=torch.tensor(
            [
                [
                    (
                        min(
                            [
                                operation.operation_id
                                for operation in job.operations
                                if operation not in job.scheduled_operations
                            ]
                        )
                        if len(job.scheduled_operations) < job.nr_of_ops
                        else job.nr_of_ops
                    )
                    for job in simulationEnv.JobShop.jobs
                ]
            ]
        ),
        end_ope_biases_batch=torch.tensor(
            [[job.operations[-1].operation_id for job in simulationEnv.JobShop.jobs]]
        ),
        time_batch=simulationEnv.simulator.now,
        nums_opes_batch=torch.tensor([simulationEnv.JobShop.nr_of_operations]),
    )

    # Get action from the dispatcher
    action = dispatcher.act(state, [], False, flag_train=False, flag_sample=False)
    operation_id = action[0, :].item()
    operation = next(
        (
            ope
            for ope in simulationEnv.JobShop.operations
            if ope.operation_id == operation_id
        ),
        None,
    )
    machine_id = action[1, :].item()
    machine = next(
        (mch for mch in simulationEnv.JobShop.machines if mch.machine_id == machine_id),
        None,
    )
    # Input action into the online scheduling process
    simulationEnv.JobShop._scheduled_operations.append(operation)
    simulationEnv.simulator.process(simulationEnv.perform_operation(operation, machine))


def load_dispatcher_FJSP_DRL():
    parser = argparse.ArgumentParser(
        description="Run (Online) Job Shop Simulation Model"
    )
    parser.add_argument(
        "-f",
        "--config_file",
        type=str,
        default=f"configs/FJSP_DRL.toml",
        help="path to config file",
    )
    args = parser.parse_args()
    param_file = args.config_file
    parameters = load_parameters(param_file)
    dispatcher = load_model(
        parameters["test_parameters"]["device"],
        parameters["model_parameters"],
        parameters["test_parameters"],
    )
    return dispatcher
