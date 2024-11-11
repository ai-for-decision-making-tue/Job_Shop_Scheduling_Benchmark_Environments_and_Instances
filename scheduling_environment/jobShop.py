import numpy as np
from typing import Dict, List

from scheduling_environment.job import Job
from scheduling_environment.machine import Machine
from scheduling_environment.operation import Operation


class JobShop:
    def __init__(self) -> None:
        self._nr_of_jobs = 0
        self._nr_of_machines = 0
        self._jobs: List[Job] = []
        self._operations: List[Operation] = []
        self._machines: List[Machine] = []
        self._precedence_relations_jobs: Dict[int, List[int]] = {}
        self._precedence_relations_operations: Dict[int, List[int]] = {}
        self._sequence_dependent_setup_times: List = []
        self._operations_to_be_scheduled: List[Operation] = []
        self._operations_available_for_scheduling: List[Operation] = []
        self._scheduled_operations: List[Operation] = []
        self._instance_name: str = ""

    def __repr__(self):
        return (
            f"<JobShop(instance={self._instance_name!r}, "
            f"jobs={self.nr_of_jobs}, operations={self.nr_of_operations}, "
            f"machines={self.nr_of_machines})>"
        )

    def reset(self):
        self._scheduled_operations = []

        self._operations_to_be_scheduled = [
            operation for operation in self._operations]

        for machine in self._machines:
            machine.reset()

        for operation in self._operations:
            operation.reset()

        self.update_operations_available_for_scheduling()

    def set_instance_name(self, name: str) -> None:
        """Set the name of the instance."""
        self._instance_name = name

    def set_nr_of_jobs(self, nr_of_jobs: int) -> None:
        """Set the number of jobs."""
        self._nr_of_jobs = nr_of_jobs

    def set_nr_of_machines(self, nr_of_machines: int) -> None:
        """Set the number of jobs."""
        self._nr_of_machines = nr_of_machines

    def add_operation(self, operation) -> None:
        """Add an operation to the environment."""
        self._operations_to_be_scheduled.append(operation)
        self._operations.append(operation)

    def add_machine(self, machine) -> None:
        """Add a machine to the environment."""
        self._machines.append(machine)

    def add_job(self, job) -> None:
        """Add a job to the environment."""
        self._jobs.append(job)

    def add_precedence_relations_jobs(self, precedence_relations_jobs: Dict[int, List[int]]) -> None:
        """Add precedence relations between jobs --> applicable for assembly scheduling problems."""
        self._precedence_relations_jobs = precedence_relations_jobs

    def add_precedence_relations_operations(self, precedence_relations_operations: Dict[int, List[int]]) -> None:
        """Add precedence relations between operations."""
        self._precedence_relations_operations = precedence_relations_operations

    def add_sequence_dependent_setup_times(self, sequence_dependent_setup_times: List) -> None:
        """Add sequence dependent setup times."""
        self._sequence_dependent_setup_times = sequence_dependent_setup_times

    def get_job(self, job_id):
        """Return operation object with operation id, or None if not found."""
        job = next((job for job in self.jobs if job.job_id == job_id), None)
        if job is None:
            raise ValueError(f"No job found with job_id: {job_id}")
        return job

    def get_operation(self, operation_id):
        """Return operation object with operation id, or None if not found."""
        operation = next((operation for operation in self.operations if operation.operation_id == operation_id), None)
        if operation is None:
            raise ValueError(f"No operation found with operation_id: {operation_id}")
        return operation

    def get_machine(self, machine_id):
        """Return machine object with machine id, or None if not found."""
        machine = next((machine for machine in self._machines if machine.machine_id == machine_id), None)
        if machine is None:
            raise ValueError(f"No machine found with machine_id: {machine_id}")
        return machine

    @property
    def jobs(self) -> List[Job]:
        """Return all the jobs."""
        return self._jobs

    @property
    def nr_of_jobs(self) -> int:
        """Return the number of jobs."""
        return self._nr_of_jobs

    @property
    def operations(self) -> List[Operation]:
        """Return all the operations."""
        return self._operations

    @property
    def nr_of_operations(self) -> int:
        """Return the number of jobs."""
        return len(self._operations)

    @property
    def machines(self) -> List[Machine]:
        """Return all the machines"""
        return self._machines

    @property
    def nr_of_machines(self) -> int:
        """Return the number of jobs."""
        return self._nr_of_machines

    @property
    def operations_to_be_scheduled(self) -> List[Operation]:
        """Return all the operations to be schedule"""
        return self._operations_to_be_scheduled

    @property
    def operations_available_for_scheduling(self) -> List[Operation]:
        """Return all the operations that are available for scheduling"""
        return self._operations_available_for_scheduling

    @property
    def scheduled_operations(self) -> List[Operation]:
        """Return all the operations that are scheduled"""
        return self._scheduled_operations

    @property
    def precedence_relations_operations(self) -> Dict[int, List[int]]:
        """Return the precedence relations between operations."""
        return self._precedence_relations_operations

    @property
    def precedence_relations_jobs(self) -> Dict[int, List[int]]:
        """Return the precedence relations between operations."""
        return self._precedence_relations_jobs

    @property
    def instance_name(self) -> str:
        """Return the name of the instance."""
        return self._instance_name

    @property
    def makespan(self) -> float:
        """Return the total makespan needed to complete all operations."""
        return max(
            [operation.scheduled_end_time for machine in self.machines for operation in machine.scheduled_operations])

    @property
    def total_workload(self) -> float:
        """Return the total workload (sum of processing times of all scheduled operations)"""
        return sum(
            [operation.scheduled_duration for machine in self.machines for operation in machine.scheduled_operations])

    @property
    def max_workload(self) -> float:
        """Return the max workload of machines (sum of processing times of all scheduled operations on a machine)"""
        return max(sum(op.scheduled_duration for op in machine.scheduled_operations) for machine in self.machines)

    @property
    def average_workload(self) -> float:
        """Return the max workload of machines (sum of processing times of all scheduled operations on a machine)"""
        return np.mean([sum(op.scheduled_duration for op in machine.scheduled_operations) for machine in self.machines])

    @property
    def balanced_workload(self) -> float:
        """Return the max workload of machines (sum of processing times of all scheduled operations on a machine)"""
        return max(sum(op.scheduled_duration for op in machine.scheduled_operations) for machine in self.machines) - min(sum(op.scheduled_duration for op in machine.scheduled_operations) for machine in self.machines)

    @property
    def average_flowtime(self) -> float:
        total_flowtime = 0
        for job in self._jobs:
            total_flowtime += job._operations[-1].scheduled_end_time - job._operations[0].scheduled_start_time
        return total_flowtime / self._nr_of_jobs

    @property
    def max_flowtime(self) -> float:
        max_flowtime = 0
        for job in self._jobs:
            flow_time = job._operations[-1].scheduled_end_time - job._operations[0].scheduled_start_time
            if flow_time > max_flowtime:
                max_flowtime = flow_time
        return max_flowtime

    def schedule_operation_on_machine(self, operation: Operation, machine_id, duration) -> None:
        """Schedule an operation on a specific machine."""
        machine = self.get_machine(machine_id)
        if machine is None:
            raise ValueError(
                f"Invalid machine ID {machine_id}")
        machine.add_operation_to_schedule(operation, duration, self._sequence_dependent_setup_times)

    def schedule_operation_with_backfilling(self, operation: Operation, machine_id, duration) -> None:
        """Schedule an operation"""
        if operation not in self.operations_available_for_scheduling:
            raise ValueError(
                f"Operation {operation.operation_id} is not available for scheduling")
        machine = self.get_machine(machine_id)
        if not machine:
            raise ValueError(
                f"Invalid machine ID {machine_id}")
        machine.add_operation_to_schedule_backfilling(operation, duration, self._sequence_dependent_setup_times)
        self.mark_operation_as_scheduled(operation)

    def unschedule_operation(self, operation: Operation) -> None:
        """Unschedule an operation"""
        machine = self.get_machine(operation.scheduled_machine)
        machine.unschedule_operation(operation)
        self.mark_operation_as_available(operation)

    def mark_operation_as_scheduled(self, operation: Operation) -> None:
        """Mark an operation as scheduled."""
        if operation not in self.operations_available_for_scheduling:
            raise ValueError(
                f"Operation {operation.operation_id} is not available for scheduling")
        self.operations_available_for_scheduling.remove(operation)
        self.scheduled_operations.append(operation)
        self.operations_to_be_scheduled.remove(operation)

    def mark_operation_as_available(self, operation: Operation) -> None:
        """Mark an operation as available for scheduling."""
        if operation not in self.scheduled_operations:
            raise ValueError(
                f"Operation {operation.operation_id} is not scheduled")
        self.scheduled_operations.remove(operation)
        self.operations_available_for_scheduling.append(operation)
        self.operations_to_be_scheduled.append(operation)

    def update_operations_available_for_scheduling(self) -> None:
        """Update the list of operations available for scheduling."""
        scheduled_operations = set(self.scheduled_operations)
        operations_available_for_scheduling = [
            operation
            for operation in self.operations
            if operation not in scheduled_operations and all(
                prec_operation in scheduled_operations
                for prec_operation in self._precedence_relations_operations[operation.operation_id]
            )
        ]
        self._operations_available_for_scheduling = operations_available_for_scheduling
