from collections import deque
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import sys

# increase recursion limit for dfs
sys.setrecursionlimit(5000)

# dfs implementation - recursive
def dfsRec(adj, visited, s, res):
    visited[s] = True
    res.append(s)
    for i in range(len(adj)):
        if adj[s][i] == 1 and not visited[i]:
            dfsRec(adj, visited, i, res)

def dfs(adj):
    visited = [False] * len(adj)
    res = []
    dfsRec(adj, visited, 0, res)
    return res

# dfs iterative (for large graphs)
def dfs_iterative(adj):
    visited = [False] * len(adj)
    res = []
    stack = [0]
    
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            res.append(v)
            
            # add neighbors to stack
            for i in range(len(adj)-1, -1, -1):
                if adj[v][i] == 1 and not visited[i]:
                    stack.append(i)
    
    return res

# bfs implementation
def bfs(adj):
    V = len(adj)
    res = []
    s = 0
    q = deque()
    
    # mark all vertices as not visited initially
    visited = [False] * V
    
    # mark source as visited
    visited[s] = True
    q.append(s)
    
    # iterate over queue
    while q:
        curr = q.popleft()
        res.append(curr)
        
        # check adjacent vertices
        for i in range(len(adj[curr])):
            if adj[curr][i] == 1 and not visited[i]:
                visited[i] = True
                q.append(i)
    
    return res

def add_edge(adj, s, t):
    adj[s][t] = 1
    adj[t][s] = 1

# graph generators - same as original
def generate_chain_graph(n):
    """chain graph 0-1-2-...-n-1"""
    adj = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        add_edge(adj, i, i + 1)
    return adj

def generate_tree(n):
    """balanced binary tree"""
    adj = [[0] * n for _ in range(n)]
    for i in range(1, n):
        parent = (i - 1) // 2
        if parent < n and i < n:
            add_edge(adj, parent, i)
    return adj

def generate_cyclic_graph(n):
    """cyclic graph - single cycle"""
    adj = generate_chain_graph(n)
    add_edge(adj, 0, n - 1)  # complete the cycle
    return adj

def generate_random_sparse_graph(n, edge_factor=2):
    """sparse graph with ~edge_factor*n edges"""
    adj = [[0] * n for _ in range(n)]
    max_edges = n * (n - 1) // 2
    target_edges = min(edge_factor * n, max_edges)
    
    edges_added = 0
    while edges_added < target_edges:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and adj[u][v] == 0:
            add_edge(adj, u, v)
            edges_added += 1
    return adj

def generate_random_dense_graph(n, density=0.5):
    """dense graph with given density"""
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < density:
                add_edge(adj, i, j)
    return adj

# performance measurement
def measure_performance(algo, graph_generator, max_size=800, step=50, repeats=30):
    sizes = range(step, max_size + 1, step)
    times = []
    
    for n in sizes:
        total_time = 0
        for _ in range(repeats):
            adj = graph_generator(n)
            
            start_time = time.time()
            algo(adj)
            end_time = time.time()
            
            total_time += (end_time - start_time)
        
        avg_time = total_time / repeats
        times.append(avg_time)
    
    return sizes, times

# main analysis function
def analyze_algorithms():
    max_size = 800  # reduced to avoid recursion limit
    step = 50
    repeats = 30  # fewer repeats for speed
    
    # all graph types from original
    graph_types = {
        "Chain Graph": generate_chain_graph,
        "Binary Tree": generate_tree,
        "Cyclic Graph": generate_cyclic_graph,
        "Random Sparse (e=2n)": lambda n: generate_random_sparse_graph(n, 2),
        "Random Dense (d=0.5)": lambda n: generate_random_dense_graph(n, 0.5)
    }
    
    # test dfs
    dfs_results = {}
    print("=== testing DFS ===")
    for name, generator in graph_types.items():
        print(f"testing {name}...")
        # use iterative dfs for chain graphs to avoid stack overflow
        algo = dfs_iterative if name == "Chain Graph" else dfs
        sizes, times = measure_performance(algo, generator, max_size, step, repeats)
        dfs_results[name] = (sizes, times)
    
    # test bfs
    bfs_results = {}
    print("\n=== testing BFS ===")
    for name, generator in graph_types.items():
        print(f"testing {name}...")
        sizes, times = measure_performance(bfs, generator, max_size, step, repeats)
        bfs_results[name] = (sizes, times)
    
    # plot results - separate graphs
    plt.figure(figsize=(15, 6))
    
    # dfs plot
    plt.subplot(1, 2, 1)
    for label, (sizes, times) in dfs_results.items():
        plt.plot(sizes, times, 'o-', label=label, markersize=4)
    plt.xlabel('number of vertices (n)')
    plt.ylabel('execution time (seconds)')
    plt.title('DFS Performance on Different Graph Types')
    plt.legend()
    plt.grid(True)
    
    # bfs plot
    plt.subplot(1, 2, 2)
    for label, (sizes, times) in bfs_results.items():
        plt.plot(sizes, times, 'o-', label=label, markersize=4)
    plt.xlabel('number of vertices (n)')
    plt.ylabel('execution time (seconds)')
    plt.title('BFS Performance on Different Graph Types')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dfs_bfs_performance.png', dpi=300)
    plt.show()
    
    # comparison plot
    plt.figure(figsize=(12, 8))
    
    # plot dfs vs bfs for each graph type
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i, name in enumerate(graph_types.keys()):
        sizes_dfs, times_dfs = dfs_results[name]
        sizes_bfs, times_bfs = bfs_results[name]
        
        plt.plot(sizes_dfs, times_dfs, 'o-', color=colors[i], label=f'DFS {name}', markersize=4)
        plt.plot(sizes_bfs, times_bfs, 's--', color=colors[i], label=f'BFS {name}', alpha=0.7, markersize=4)
    
    plt.xlabel('number of vertices (n)')
    plt.ylabel('execution time (seconds)')
    plt.title('DFS vs BFS Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dfs_bfs_comparison.png', dpi=300)
    plt.show()
    
    # ratio plot - dfs/bfs time
    plt.figure(figsize=(10, 6))
    
    for name in graph_types.keys():
        sizes_dfs, times_dfs = dfs_results[name]
        sizes_bfs, times_bfs = bfs_results[name]
        
        # calculate ratios
        ratios = [t_dfs/t_bfs if t_bfs > 0 else 1 for t_dfs, t_bfs in zip(times_dfs, times_bfs)]
        
        plt.plot(sizes_dfs, ratios, 'o-', label=name, markersize=4)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('number of vertices (n)')
    plt.ylabel('time ratio (DFS/BFS)')
    plt.title('Performance Ratio: DFS/BFS')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dfs_bfs_ratio.png', dpi=300)
    plt.show()
    
    # print some statistics
    print("\n=== Performance Summary ===")
    for graph_name in graph_types.keys():
        dfs_sizes, dfs_times = dfs_results[graph_name]
        bfs_sizes, bfs_times = bfs_results[graph_name]
        
        # compare at largest size
        final_dfs = dfs_times[-1]
        final_bfs = bfs_times[-1]
        
        print(f"\n{graph_name} (n={dfs_sizes[-1]}):")
        print(f"  DFS: {final_dfs:.6f}s")
        print(f"  BFS: {final_bfs:.6f}s")
        print(f"  Ratio (DFS/BFS): {final_dfs/final_bfs:.3f}")

# run the analysis
if __name__ == "__main__":
    analyze_algorithms()
