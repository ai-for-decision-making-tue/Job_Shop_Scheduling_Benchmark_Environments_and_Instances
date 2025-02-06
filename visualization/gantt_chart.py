import matplotlib.pyplot as plt
from visualization.color_scheme import create_colormap


def plot(JobShop):
    # Plot the Gantt chart of the job shop schedule
    fig, ax = plt.subplots()
    colormap = create_colormap()

    for machine in JobShop.machines:
        machine_operations = sorted(machine._processed_operations, key=lambda op: op.scheduling_information['start_time'])
        for operation in machine_operations:
            operation_start = operation.scheduling_information['start_time']
            operation_end = operation.scheduling_information['end_time']
            operation_duration = operation_end - operation_start
            operation_label = f"{operation.operation_id}"

            # Set color based on job ID
            color_index = operation.job_id % len(JobShop.jobs)
            if color_index >= colormap.N:
                color_index = color_index % colormap.N
            color = colormap(color_index)

            ax.broken_barh(
                [(operation_start, operation_duration)],
                (machine.machine_id - 0.4, 0.8),
                facecolors=color,
                edgecolor='black'
            )

            setup_start = operation.scheduling_information['start_setup']
            setup_time = operation.scheduling_information['setup_time']
            if setup_time != None:
                ax.broken_barh(
                    [(setup_start, setup_time)],
                    (machine.machine_id - 0.4, 0.8),
                    facecolors='grey',
                    edgecolor='black', hatch='/')
            middle_of_operation = operation_start + operation_duration / 2
            ax.text(
                middle_of_operation,
                machine.machine_id,
                operation_label,
                ha='center',
                va='center',
                fontsize=8
            )

    fig = ax.figure
    fig.set_size_inches(12, 6)

    ax.set_yticks(range(JobShop.nr_of_machines))
    ax.set_yticklabels([f'M{machine_id+1}' for machine_id in range(JobShop.nr_of_machines)])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('Gantt Chart')
    ax.grid(True)

    return plt
