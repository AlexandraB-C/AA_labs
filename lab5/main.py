import time
import random
import heapq
import matplotlib.pyplot as plt
import numpy as np
from math import inf

# --------------------------
# Prim's Algorithm
# --------------------------
def is_symmetric(graph):
    n = len(graph)
    for i in range(n):
        for j in range(n):
            if graph[i][j] != graph[j][i]:
                return False
    return True

def prim(graph):
    if not is_symmetric(graph):
        raise ValueError("Graph is directed. MST does not exist.")

    n = len(graph)
    visited = [False] * n
    min_edge = [inf] * n
    min_edge[0] = 0
    heap = [(0, 0)]
    count = 0

    while heap:
        weight, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        count += 1

        for v in range(n):
            if not visited[v] and graph[u][v] < min_edge[v]:
                min_edge[v] = graph[u][v]
                heapq.heappush(heap, (graph[u][v], v))

    if count < n:
        raise ValueError("Graph is disconnected. MST does not exist.")



# --------------------------
# Graph Generation Functions
# --------------------------
def generate_sparse_graph(n):
    graph = [[inf for _ in range(n)] for _ in range(n)]
    for i in range(n):
        neighbors = random.sample(range(n), min(4, max(1, n - 1)))
        for j in neighbors:
            if i != j:
                weight = random.randint(1, 10)
                graph[i][j] = weight
                graph[j][i] = weight
    return graph

def generate_dense_graph(n):
    graph = [[inf for _ in range(n)] for _ in range(n)]
    for i in range(n):
        neighbors = random.sample(range(n), max(1, int(n * 0.2)))
        for j in neighbors:
            if i != j:
                weight = random.randint(1, 10)
                graph[i][j] = weight
                graph[j][i] = weight
    return graph

def generate_undirected_graph(n):
    return generate_sparse_graph(n)

def generate_directed_graph(n):
    graph = [[inf for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and random.random() < 0.05:  # lower chance
                weight = random.randint(1, 10)
                graph[i][j] = weight  # Only one direction!
    return graph


def generate_weighted_graph(n):
    return generate_sparse_graph(n)

def generate_unweighted_graph(n):
    graph = [[inf for _ in range(n)] for _ in range(n)]
    for i in range(n):
        neighbors = random.sample(range(n), min(4, max(1, n - 1)))
        for j in neighbors:
            if i != j:
                graph[i][j] = 1
                graph[j][i] = 1
    return graph

def generate_connected_graph(n):
    return generate_sparse_graph(n)

def generate_disconnected_graph(n):
    graph = [[inf for _ in range(n)] for _ in range(n)]
    for i in range(n // 2 - 1):
        weight = random.randint(1, 10)
        graph[i][i + 1] = weight
        graph[i + 1][i] = weight
    for i in range(n // 2, n - 1):
        weight = random.randint(1, 10)
        graph[i][i + 1] = weight
        graph[i + 1][i] = weight
    return graph

def generate_cyclic_graph(n):
    graph = generate_sparse_graph(n)
    extra_edges = n // 2
    for _ in range(extra_edges):
        u, v = random.sample(range(n), 2)
        weight = random.randint(1, 10)
        graph[u][v] = weight
        graph[v][u] = weight
    return graph

def generate_acyclic_graph(n):
    graph = [[inf for _ in range(n)] for _ in range(n)]
    for i in range(n - 1):
        weight = random.randint(1, 10)
        graph[i][i + 1] = weight
        graph[i + 1][i] = weight
    return graph

def generate_complete_graph(n):
    graph = [[inf for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            weight = random.randint(1, 10)
            graph[i][j] = weight
            graph[j][i] = weight
    return graph

def generate_tree_graph(n):
    graph = [[inf for _ in range(n)] for _ in range(n)]
    for i in range(1, n):
        j = random.randint(0, i - 1)
        weight = random.randint(1, 10)
        graph[i][j] = weight
        graph[j][i] = weight
    return graph

# --------------------------
# Empirical Analysis
# --------------------------
def run_experiment(graph_generator, max_nodes, step, num_trials=3):
    node_counts = list(range(10, max_nodes + 1, step))
    times = []

    for n in node_counts:
        total_time = 0
        successful_trials = 0

        for _ in range(num_trials):
            graph = graph_generator(n)
            try:
                start_time = time.time()
                prim(graph)
                total_time += time.time() - start_time
                successful_trials += 1
            except ValueError as e:
                # Skip invalid graphs (e.g., disconnected)
                continue

        if successful_trials > 0:
            times.append(total_time / successful_trials)
        else:
            # Use None or 0 to indicate failure
            times.append(None)

    return node_counts, times


# --------------------------
# Plotting Results
# --------------------------
def plot_all_results(results_dict, excluded_labels):
    plt.figure(figsize=(10, 7))

    for label, (x, y) in results_dict.items():
        if label in excluded_labels:
            plt.plot([], [], marker='o', label=f"{label} (excluded)")
        else:
            y_fixed = [val if val is not None else np.nan for val in y]
            plt.plot(x, y_fixed, marker='o', label=label)

    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (seconds)')
    plt.title("Time Complexity of Prim's Algorithm on Different Graph Types")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# --------------------------
# Main Execution
# --------------------------
graph_generators = {
    "Undirected": generate_undirected_graph,
    "Directed": generate_directed_graph,
    "Weighted": generate_weighted_graph,
    "Unweighted": generate_unweighted_graph,
    "Connected": generate_connected_graph,
    "Disconnected": generate_disconnected_graph,
    "Cyclic": generate_cyclic_graph,
    "Acyclic": generate_acyclic_graph,
    "Complete": generate_complete_graph,
    "Sparse": generate_sparse_graph,
    "Dense": generate_dense_graph,
    "Tree": generate_tree_graph
}

max_nodes = 300
step = 10
num_trials = 3

results = {}
excluded = []

for label, generator in graph_generators.items():
    print(f"Running on {label} graphs...")
    x, y = run_experiment(generator, max_nodes=max_nodes, step=step, num_trials=num_trials)

    # If all values are None, we assume it's invalid (Prim cannot build MST)
    if all(v is None for v in y):
        print(f"  ⚠️ Skipping {label} from plot (Prim's not applicable).")
        excluded.append(label)
        results[label] = ([], [])  # Placeholder for legend
    else:
        results[label] = (x, y)


# Plot
plot_all_results(results, excluded)
