from .operation import Operation
from typing import List


class Job:
    def __init__(self, job_id: int):
        self._job_id: int = job_id
        self._operations: List[Operation] = []

    def add_operation(self, operation: Operation):
        """Add an operation to the job."""
        self._operations.append(operation)

    @property
    def nr_of_ops(self) -> int:
        """Return the number of jobs."""
        return len(self._operations)

    @property
    def operations(self) -> List[Operation]:
        """Return the list of operations."""
        return self._operations

    @property
    def job_id(self) -> int:
        """Return the job's id."""
        return self._job_id

    def get_operation(self, operation_id):
        """Return operation object with operation id."""
        for operation in self._operations:
            if operation.operation_id == operation_id:
                return operation
