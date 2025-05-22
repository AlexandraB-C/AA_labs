import time
import random
import matplotlib.pyplot as plt
import numpy as np
from math import inf
import heapq


# --------------------------
# Union-Find for Kruskal's Algorithm
# --------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # path compression
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u == root_v:
            return False
        # union by rank
        if self.rank[root_u] < self.rank[root_v]:
            self.parent[root_u] = root_v
        else:
            self.parent[root_v] = root_u
            if self.rank[root_u] == self.rank[root_v]:
                self.rank[root_u] += 1
        return True


# --------------------------
# Utility Functions
# --------------------------
def is_symmetric(graph):
    """check if graph is undirected (symmetric)"""
    n = len(graph)
    for i in range(n):
        for j in range(n):
            if graph[i][j] != graph[j][i]:
                return False
    return True


def is_connected(graph):
    """check if graph is connected using dfs"""
    n = len(graph)
    visited = [False] * n
    
    def dfs(v):
        visited[v] = True
        for u in range(n):
            if graph[v][u] != inf and not visited[u]:
                dfs(u)
    
    dfs(0)
    return all(visited)


# --------------------------
# Kruskal's Algorithm
# --------------------------
def kruskal(graph):
    """kruskal's mst algorithm - edge-based greedy"""
    if not is_symmetric(graph):
        raise ValueError("graph is directed. mst does not exist.")
    
    n = len(graph)
    edges = []
    
    # collect all edges
    for i in range(n):
        for j in range(i + 1, n):
            if graph[i][j] != inf:
                edges.append((graph[i][j], i, j))
    
    if len(edges) == 0:
        raise ValueError("graph has no edges.")
    
    # sort edges by weight (greedy choice)
    edges.sort()
    uf = UnionFind(n)
    mst_edges = []
    total_weight = 0
    
    for weight, u, v in edges:
        if uf.union(u, v):  # no cycle created
            mst_edges.append((u, v, weight))
            total_weight += weight
            if len(mst_edges) == n - 1:
                break
    
    if len(mst_edges) < n - 1:
        raise ValueError("graph is disconnected. mst does not exist.")
    
    return mst_edges, total_weight


# --------------------------
# Prim's Algorithm
# --------------------------
def prim(graph):
    """prim's mst algorithm - vertex-based greedy"""
    if not is_symmetric(graph):
        raise ValueError("graph is directed. mst does not exist.")
    
    n = len(graph)
    if n == 0:
        return [], 0
    
    # check connectivity
    if not is_connected(graph):
        raise ValueError("graph is disconnected. mst does not exist.")
    
    visited = [False] * n
    mst_edges = []
    total_weight = 0
    
    # start from vertex 0
    visited[0] = True
    pq = []  # priority queue (weight, from_vertex, to_vertex)
    
    # add all edges from vertex 0
    for j in range(n):
        if graph[0][j] != inf:
            heapq.heappush(pq, (graph[0][j], 0, j))
    
    while pq and len(mst_edges) < n - 1:
        weight, u, v = heapq.heappop(pq)
        
        if visited[v]:  # already in mst
            continue
        
        # add to mst (greedy choice)
        visited[v] = True
        mst_edges.append((u, v, weight))
        total_weight += weight
        
        # add new edges from v
        for j in range(n):
            if graph[v][j] != inf and not visited[j]:
                heapq.heappush(pq, (graph[v][j], v, j))
    
    if len(mst_edges) < n - 1:
        raise ValueError("graph is disconnected. mst does not exist.")
    
    return mst_edges, total_weight


# --------------------------
# Graph Generation Functions
# --------------------------
def generate_sparse_graph(n):
    """sparse graph ~30% edges"""
    graph = [[inf for _ in range(n)] for _ in range(n)]
    for i in range(n):
        # ensure connectivity first
        if i > 0:
            j = random.randint(0, i - 1)
            weight = random.randint(1, 10)
            graph[i][j] = weight
            graph[j][i] = weight
        
        # add sparse connections
        neighbors = random.sample(range(n), min(3, max(1, n - 1)))
        for j in neighbors:
            if i != j:
                weight = random.randint(1, 10)
                graph[i][j] = weight
                graph[j][i] = weight
    return graph


def generate_dense_graph(n):
    """dense graph ~80% edges"""
    graph = [[inf for _ in range(n)] for _ in range(n)]
    # ensure connectivity
    for i in range(1, n):
        j = random.randint(0, i - 1)
        weight = random.randint(1, 10)
        graph[i][j] = weight
        graph[j][i] = weight
    
    # add dense connections
    for i in range(n):
        neighbors = random.sample(range(n), max(1, int(n * 0.8)))
        for j in neighbors:
            if i != j:
                weight = random.randint(1, 10)
                graph[i][j] = weight
                graph[j][i] = weight
    return graph


def generate_undirected_graph(n):
    return generate_sparse_graph(n)


def generate_directed_graph(n):
    """directed graph - not suitable for mst"""
    graph = [[inf for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and random.random() < 0.1:
                weight = random.randint(1, 10)
                graph[i][j] = weight  # only one direction
    return graph


def generate_weighted_graph(n):
    return generate_sparse_graph(n)


def generate_unweighted_graph(n):
    """unweighted graph - all edges weight 1"""
    graph = [[inf for _ in range(n)] for _ in range(n)]
    # ensure connectivity
    for i in range(1, n):
        j = random.randint(0, i - 1)
        graph[i][j] = 1
        graph[j][i] = 1
    
    # add more edges
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
    """disconnected graph - two components"""
    graph = [[inf for _ in range(n)] for _ in range(n)]
    mid = n // 2
    
    # first component
    for i in range(mid - 1):
        weight = random.randint(1, 10)
        graph[i][i + 1] = weight
        graph[i + 1][i] = weight
    
    # second component
    for i in range(mid, n - 1):
        weight = random.randint(1, 10)
        graph[i][i + 1] = weight
        graph[i + 1][i] = weight
    
    return graph


def generate_cyclic_graph(n):
    """graph with cycles"""
    graph = generate_sparse_graph(n)
    # add extra edges to create cycles
    extra = n // 3
    for _ in range(extra):
        u, v = random.sample(range(n), 2)
        weight = random.randint(1, 10)
        graph[u][v] = weight
        graph[v][u] = weight
    return graph


def generate_acyclic_graph(n):
    """tree structure - no cycles"""
    graph = [[inf for _ in range(n)] for _ in range(n)]
    for i in range(1, n):
        j = random.randint(0, i - 1)
        weight = random.randint(1, 10)
        graph[i][j] = weight
        graph[j][i] = weight
    return graph


def generate_complete_graph(n):
    """complete graph - all possible edges"""
    graph = [[inf for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            weight = random.randint(1, 10)
            graph[i][j] = weight
            graph[j][i] = weight
    return graph


def generate_tree_graph(n):
    """tree structure"""
    return generate_acyclic_graph(n)


# --------------------------
# Empirical Analysis
# --------------------------
def run_experiment_both(graph_generator, max_nodes, step, num_trials=3):
    """run both prim and kruskal on same graphs"""
    node_counts = list(range(10, max_nodes + 1, step))
    prim_times = []
    kruskal_times = []
    
    for n in node_counts:
        prim_total = 0
        kruskal_total = 0
        successful_trials = 0
        
        for _ in range(num_trials):
            graph = graph_generator(n)
            try:
                # test prim
                start_time = time.time()
                prim(graph)
                prim_total += time.time() - start_time
                
                # test kruskal on same graph
                start_time = time.time()
                kruskal(graph)
                kruskal_total += time.time() - start_time
                
                successful_trials += 1
            except ValueError:
                continue
        
        if successful_trials > 0:
            prim_times.append(prim_total / successful_trials)
            kruskal_times.append(kruskal_total / successful_trials)
        else:
            prim_times.append(None)
            kruskal_times.append(None)
    
    return node_counts, prim_times, kruskal_times


def run_single_experiment(algorithm, graph_generator, max_nodes, step, num_trials=3):
    """run single algorithm experiment"""
    node_counts = list(range(10, max_nodes + 1, step))
    times = []
    
    for n in node_counts:
        total_time = 0
        successful_trials = 0
        
        for _ in range(num_trials):
            graph = graph_generator(n)
            try:
                start_time = time.time()
                algorithm(graph)
                total_time += time.time() - start_time
                successful_trials += 1
            except ValueError:
                continue
        
        if successful_trials > 0:
            times.append(total_time / successful_trials)
        else:
            times.append(None)
    
    return node_counts, times


# --------------------------
# Plotting Results
# --------------------------
def plot_comparison_results(results_dict, excluded_labels):
    """plot comparison between prim and kruskal"""
    plt.figure(figsize=(15, 10))
    
    # plot 1: sparse vs dense comparison
    plt.subplot(2, 2, 1)
    if 'Sparse' in results_dict and 'Sparse' not in excluded_labels:
        x, prim_y, kruskal_y = results_dict['Sparse']
        prim_fixed = [val if val is not None else np.nan for val in prim_y]
        kruskal_fixed = [val if val is not None else np.nan for val in kruskal_y]
        plt.plot(x, prim_fixed, 'b-o', label='prim sparse', linewidth=2)
        plt.plot(x, kruskal_fixed, 'r-s', label='kruskal sparse', linewidth=2)
    
    if 'Dense' in results_dict and 'Dense' not in excluded_labels:
        x, prim_y, kruskal_y = results_dict['Dense']
        prim_fixed = [val if val is not None else np.nan for val in prim_y]
        kruskal_fixed = [val if val is not None else np.nan for val in kruskal_y]
        plt.plot(x, prim_fixed, 'g-^', label='prim dense', linewidth=2)
        plt.plot(x, kruskal_fixed, 'm-d', label='kruskal dense', linewidth=2)
    
    plt.xlabel('nodes')
    plt.ylabel('time (sec)')
    plt.title('sparse vs dense comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # plot 2: complete vs tree comparison
    plt.subplot(2, 2, 2)
    for graph_type in ['Complete', 'Tree']:
        if graph_type in results_dict and graph_type not in excluded_labels:
            x, prim_y, kruskal_y = results_dict[graph_type]
            prim_fixed = [val if val is not None else np.nan for val in prim_y]
            kruskal_fixed = [val if val is not None else np.nan for val in kruskal_y]
            plt.plot(x, prim_fixed, '-o', label=f'prim {graph_type.lower()}', linewidth=2)
            plt.plot(x, kruskal_fixed, '-s', label=f'kruskal {graph_type.lower()}', linewidth=2)
    
    plt.xlabel('nodes')
    plt.ylabel('time (sec)')
    plt.title('complete vs tree comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # plot 3: efficiency ratio
    plt.subplot(2, 2, 3)
    for graph_type in ['Sparse', 'Dense', 'Complete']:
        if graph_type in results_dict and graph_type not in excluded_labels:
            x, prim_y, kruskal_y = results_dict[graph_type]
            ratios = []
            for p, k in zip(prim_y, kruskal_y):
                if p is not None and k is not None and p > 0:
                    ratios.append(k / p)
                else:
                    ratios.append(np.nan)
            plt.plot(x, ratios, '-o', label=f'{graph_type.lower()} ratio', linewidth=2)
    
    plt.xlabel('nodes')
    plt.ylabel('kruskal/prim ratio')
    plt.title('efficiency ratio analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # plot 4: all graph types overview
    plt.subplot(2, 2, 4)
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    color_idx = 0
    
    for label, data in results_dict.items():
        if label not in excluded_labels:
            x, prim_y, kruskal_y = data
            color = colors[color_idx % len(colors)]
            prim_fixed = [val if val is not None else np.nan for val in prim_y]
            plt.plot(x, prim_fixed, f'{color}-', label=f'prim {label.lower()}', alpha=0.7)
            color_idx += 1
    
    plt.xlabel('nodes')
    plt.ylabel('time (sec)')
    plt.title('prim algorithm - all graph types')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mst_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# --------------------------
# Main Execution
# --------------------------
def main():
    print("greedy algorithms (mst) comprehensive analysis")
    print("=" * 60)
    
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
    
    max_nodes = 100
    step = 10
    num_trials = 3
    results = {}
    excluded = []
    
    for label, generator in graph_generators.items():
        print(f"running on {label} graphs...")
        
        try:
            if label in ["Directed", "Disconnected"]:
                # these will likely fail, test with single algorithm
                x, times = run_single_experiment(kruskal, generator, max_nodes, step, num_trials)
                if all(v is None for v in times):
                    print(f"  ⚠️ skipping {label} (not suitable for mst)")
                    excluded.append(label)
                    results[label] = ([], [], [])
                else:
                    results[label] = (x, times, times)  # dummy for plotting
            else:
                x, prim_times, kruskal_times = run_experiment_both(generator, max_nodes, step, num_trials)
                if all(v is None for v in prim_times) and all(v is None for v in kruskal_times):
                    print(f"  ⚠️ skipping {label} (not suitable for mst)")
                    excluded.append(label)
                    results[label] = ([], [], [])
                else:
                    results[label] = (x, prim_times, kruskal_times)
                    
        except Exception as e:
            print(f"  ⚠️ error with {label}: {e}")
            excluded.append(label)
            results[label] = ([], [], [])
    
    # create visualizations
    plot_comparison_results(results, excluded)
    
    # print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for label, data in results.items():
        if label not in excluded and len(data[1]) > 0:
            x, prim_times, kruskal_times = data
            valid_prim = [t for t in prim_times if t is not None]
            valid_kruskal = [t for t in kruskal_times if t is not None]
            
            if valid_prim and valid_kruskal:
                avg_prim = sum(valid_prim) / len(valid_prim)
                avg_kruskal = sum(valid_kruskal) / len(valid_kruskal)
                ratio = avg_kruskal / avg_prim if avg_prim > 0 else 0
                print(f"{label:<12}: prim {avg_prim:.4f}s, kruskal {avg_kruskal:.4f}s, ratio {ratio:.2f}x")
    
    print("\nanalysis complete! check 'mst_comparison.png'")


if __name__ == "__main__":
    main()