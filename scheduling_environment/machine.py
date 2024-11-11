from typing import List

from scheduling_environment.operation import Operation


class Machine:
    def __init__(self, machine_id, machine_name=None):
        self._machine_id = machine_id
        self._machine_name = machine_name
        self._processed_operations = []

    def __str__(self):
        return f"Machine {self._machine_id}, {len(self._processed_operations)} scheduled operations"

    def reset(self):
        self._processed_operations = []

    @property
    def machine_id(self):
        """Return the machine's id."""
        return self._machine_id

    @property
    def machine_name(self):
        """Return the machine's name."""
        return self._machine_name

    @property
    def scheduled_operations(self) -> List[Operation]:
        """Return the list of scheduled operations on this machine."""
        sorted_operations = sorted(self._processed_operations, key=lambda op: op.scheduling_information['start_time'])
        return [op for op in sorted_operations]

    @property
    def next_available_time(self):
        """Returns the time moment all currently scheduled operations have been finished on this machine."""
        return max([operation.scheduled_end_time for operation in self.scheduled_operations], default=0)

    def add_operation_to_schedule(self, operation: Operation, processing_time, sequence_dependent_setup_times):
        """Add an operation to the scheduled operations list without backfilling at earliest possible time."""

        # find max finishing time predecessors
        finishing_time_predecessors = operation.finishing_time_predecessors
        finishing_time_machine = max([operation.scheduled_end_time for operation in self.scheduled_operations],default=0)

        setup_time = 0
        if len(self.scheduled_operations) != 0:
            setup_time = \
                sequence_dependent_setup_times[self.machine_id][self.scheduled_operations[-1].operation_id][
                    operation.operation_id]
        start_time = max(finishing_time_predecessors, finishing_time_machine + setup_time)
        operation.add_operation_scheduling_information(self.machine_id, start_time, setup_time, processing_time)

        self._processed_operations.append(operation)

    def add_operation_to_schedule_at_time(self, operation, start_time, processing_time, setup_time):
        """Scheduled an operations at a certain time."""
        operation.add_operation_scheduling_information(
            self.machine_id, start_time, setup_time, processing_time)

        self._processed_operations.append(operation)

    def add_operation_to_schedule_backfilling(self, operation: Operation, processing_time, sequence_dependent_setup_times):
        """Add an operation to the scheduled operations list of the machine using backfilling (if possible)."""

        # find max finishing time predecessors
        finishing_time_predecessors = operation.finishing_time_predecessors
        finishing_time_machine = max([operation.scheduled_end_time for operation in self.scheduled_operations],
                                     default=0)

        setup_time = 0
        if len(self.scheduled_operations) != 0:
            setup_time = \
                sequence_dependent_setup_times[self.machine_id][self.scheduled_operations[-1].operation_id][
                    operation.operation_id]

        # # find backfilling opportunity
        start_time_backfilling, setup_time_backfilling = self.find_backfilling_opportunity(
            operation, finishing_time_predecessors, processing_time, sequence_dependent_setup_times)

        if start_time_backfilling is not None:
            start_time = start_time_backfilling
            setup_time = setup_time_backfilling
        else:
            # find time when predecessors are finished and machine is available
            start_time = max(finishing_time_predecessors,
                             finishing_time_machine + setup_time)

        operation.add_operation_scheduling_information(
            self.machine_id, start_time, setup_time, processing_time)

        self._processed_operations.append(operation)

    def find_backfilling_opportunity(self, operation, finishing_time_predecessors, duration,
                                     sequence_dependent_setup_times):
        """Find opportunity to earliest time to start the operation on this machine."""
        if not self.scheduled_operations:
            return None, None

        # Check for backfilling opportunity before the first scheduled operation
        first_op = self.scheduled_operations[0]
        setup_to_first = sequence_dependent_setup_times[self.machine_id][operation.operation_id][first_op.operation_id]
        if (
                duration <= first_op.scheduled_start_time - setup_to_first
                and finishing_time_predecessors <= first_op.scheduled_start_time - duration - setup_to_first
        ):
            start_time = min(finishing_time_predecessors, first_op.scheduled_start_time - duration - setup_to_first)
            # Update setup time for the first operation
            first_op.update_scheduled_sequence_dependent_setup_times(first_op.scheduled_start_time - setup_to_first,
                                                                     setup_to_first)
            return start_time, 0

        # Check for gaps between scheduled operations
        for i in range(1, len(self.scheduled_operations)):
            prev_op = self.scheduled_operations[i - 1]
            next_op = self.scheduled_operations[i]
            setup_to_prev = sequence_dependent_setup_times[self.machine_id][prev_op.operation_id][
                operation.operation_id]
            setup_to_next = sequence_dependent_setup_times[self.machine_id][operation.operation_id][
                next_op.operation_id]

            # Calculate gap duration between prev_op and next_op
            gap_duration = next_op.scheduled_start_time - prev_op.scheduled_end_time
            # Check if gap is large enough for backfilling (processing time + updated setup times)
            if gap_duration >= duration + setup_to_prev + setup_to_next:
                # Check if gap is large enough if predecessor finishes within the gap duration
                gap_start_time = max(finishing_time_predecessors, prev_op.scheduled_end_time + setup_to_prev)
                if gap_start_time + duration + setup_to_next <= next_op.scheduled_start_time:
                    # Update setup time for next operation
                    next_op.update_scheduled_sequence_dependent_setup_times(
                        next_op.scheduled_start_time - setup_to_next, setup_to_next
                    )
                    return gap_start_time, setup_to_prev

        return None, None

    def unschedule_operation(self, operation: Operation):
        """Remove an operation from the scheduled operations list of the machine."""
        self._processed_operations.remove(operation)
