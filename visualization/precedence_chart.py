import copy
import networkx as nx
import matplotlib.pyplot as plt

from visualization.color_scheme import create_colormap


def plot(JobShop):
    colormap = create_colormap()

    # Convert precedence relations into a usable format
    precedence_relations = copy.deepcopy(JobShop.precedence_relations_operations)
    for key, value in precedence_relations.items():
        value = [i.operation_id for i in value]
        precedence_relations[key] = value

    # Add nodes and edges to the graph
    G = nx.DiGraph()
    for key, value in precedence_relations.items():
        for successor in value:
            G.add_edge(successor, key)  # Reverse the edge direction

    # Assign levels to nodes using breadth-first search (BFS)
    levels = {}
    queue = [node for node in G.nodes if not list(G.predecessors(node))]
    for node in queue:
        levels[node] = 0

    while queue:
        current = queue.pop(0)
        for successor in G.successors(current):
            levels[successor] = max(levels.get(successor, 0), levels[current] + 1)
            queue.append(successor)

    # Group nodes by their levels
    level_nodes = {}
    for node, level in levels.items():
        level_nodes.setdefault(level, []).append(node)

    # Set positions for nodes
    pos = {}
    for level, nodes in level_nodes.items():
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (level, i - len(nodes) / 2)

    # Draw the graph
    options = {
        "font_size": 8,
        "node_size": 500,
        "node_color": [],
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }

    # Assign colors to nodes based on job ID
    for node in G.nodes:
        job_id = JobShop.get_operation(node).job_id
        options["node_color"].append(colormap(job_id % colormap.N))

    nx.draw_networkx(G, pos, **options)

    # Adjust plot settings
    plt.gca().margins(0.20)
    plt.gcf().set_size_inches(16, 8)
    plt.axis("off")
    plt.show()
