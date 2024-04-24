from typing import List

from scheduling_environment.operation import Operation


class Machine:
    def __init__(self, machine_id, machine_name=None):
        self._machine_id = machine_id
        self._machine_name = machine_name
        self._processed_operations = []

    def reset(self):
        self._processed_operations = []

    def __str__(self):
        return f"Machine {self._machine_id}, {len(self._processed_operations)} scheduled operations"

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
        """Add an operation to the scheduled operations list of the machine without backfilling."""

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
        """Add an operation to the scheduled operations list of the machine using backfilling."""

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
        """Find the earliest time to start the operation on this machine."""

        if len(self.scheduled_operations) > 0:
            # check if backfilling is possible before the first scheduled operation:
            if duration <= self.scheduled_operations[0].scheduled_start_time - sequence_dependent_setup_times[self.machine_id][operation.operation_id][self.scheduled_operations[0].operation_id] \
                    and finishing_time_predecessors <= self.scheduled_operations[0].scheduled_start_time - duration - sequence_dependent_setup_times[self.machine_id][operation.operation_id][self.scheduled_operations[0].operation_id]:
                start_time_backfilling = min([finishing_time_predecessors, (self.scheduled_operations[0].scheduled_start_time - duration -
                                             sequence_dependent_setup_times[self.machine_id][operation.operation_id][self.scheduled_operations[0].operation_id])])

                # update sequence dependent setup time for next operation on machine
                next_operation = self.scheduled_operations[0]
                next_operation.update_sequence_dependent_setup_times(next_operation.scheduled_start_time - sequence_dependent_setup_times[self.machine_id][operation.operation_id][
                                                                     self.scheduled_operations[0].operation_id], sequence_dependent_setup_times[self.machine_id][operation.operation_id][self.scheduled_operations[0].operation_id])
                # assumption that the first operation has no setup times!
                return start_time_backfilling, 0

            else:
                for i in range(1, len(self.scheduled_operations[1:])):
                    # if gap between two operations is large enough to fit the new operations (including setup times)
                    if (self.scheduled_operations[i].scheduled_start_time - self.scheduled_operations[i - 1].scheduled_end_time) >= duration + sequence_dependent_setup_times[self.machine_id][self.scheduled_operations[i - 1].operation_id][operation.operation_id] + sequence_dependent_setup_times[self.machine_id][operation.operation_id][self.scheduled_operations[i].operation_id]:

                        # if predecessors finishes before a potential start time
                        if self.scheduled_operations[i - 1].scheduled_end_time + sequence_dependent_setup_times[self.machine_id][self.scheduled_operations[i - 1].operation_id][operation.operation_id] >= finishing_time_predecessors:
                            # update sequence dependent setup time for next operation on machine
                            self.scheduled_operations[i].update_sequence_dependent_setup_times(
                                self.scheduled_operations[i].scheduled_start_time -
                                sequence_dependent_setup_times[self.machine_id][operation.operation_id][
                                    self.scheduled_operations[i].operation_id],
                                sequence_dependent_setup_times[self.machine_id][operation.operation_id][
                                    self.scheduled_operations[i].operation_id])
                            return self.scheduled_operations[i - 1].scheduled_end_time + sequence_dependent_setup_times[self.machine_id][self.scheduled_operations[i - 1].operation_id][operation.operation_id], sequence_dependent_setup_times[self.machine_id][self.scheduled_operations[i - 1].operation_id][operation.operation_id]
                        elif finishing_time_predecessors + sequence_dependent_setup_times[self.machine_id][operation.operation_id][self.scheduled_operations[i].operation_id] + duration <= self.scheduled_operations[i - 1].scheduled_end_time:
                            self.scheduled_operations[i].update_sequence_dependent_setup_times(
                                self.scheduled_operations[i].scheduled_start_time -
                                sequence_dependent_setup_times[self.machine_id][operation.operation_id][
                                    self.scheduled_operations[i].operation_id],
                                sequence_dependent_setup_times[self.machine_id][operation.operation_id][
                                    self.scheduled_operations[i].operation_id])

                            return finishing_time_predecessors + sequence_dependent_setup_times[self.machine_id][self.scheduled_operations[i - 1].operation_id][operation.operation_id], sequence_dependent_setup_times[self.machine_id][self.scheduled_operations[i - 1].operation_id][operation.operation_id]

        return None, None

    def unschedule_operation(self, operation: Operation):
        """Remove an operation from the scheduled operations list of the machine."""
        self._processed_operations.remove(operation)
