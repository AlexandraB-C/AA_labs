import time
import matplotlib.pyplot as plt
from collections import deque
import random

class Graph:
    def __init__(self):
        self.graph = {}
    
    def add_edge(self, u, v):
        # add edge between nodes
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        self.graph[u].append(v)
        self.graph[v].append(u)  # undirected graph
    
    def dfs(self, start, target):
        # depth first search
        visited = set()
        stack = [start]
        operations = 0
        
        while stack:
            operations += 1
            node = stack.pop()
            
            if node == target:
                return operations, True
            
            if node not in visited:
                visited.add(node)
                # add neighbors to stack
                for neighbor in self.graph.get(node, []):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return operations, False
    
    def bfs(self, start, target):
        # breadth first search
        visited = set()
        queue = deque([start])
        visited.add(start)
        operations = 0
        
        while queue:
            operations += 1
            node = queue.popleft()
            
            if node == target:
                return operations, True
            
            # add neighbors to queue
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return operations, False

def create_chain_graph(nodes):
    # chain graph: 0-1-2-3-...-n
    g = Graph()
    for i in range(nodes-1):
        g.add_edge(i, i+1)
    return g

def create_binary_tree(nodes):
    # binary tree structure
    g = Graph()
    for i in range(nodes//2):
        left_child = 2*i + 1
        right_child = 2*i + 2
        if left_child < nodes:
            g.add_edge(i, left_child)
        if right_child < nodes:
            g.add_edge(i, right_child)
    return g

def create_cyclic_graph(nodes):
    # cycle graph: 0-1-2-...-n-0
    g = Graph()
    for i in range(nodes):
        g.add_edge(i, (i+1) % nodes)
    return g

def create_random_sparse(nodes):
    # random sparse graph (e=2n)
    g = Graph()
    edges_count = 2 * nodes
    for _ in range(edges_count):
        u = random.randint(0, nodes-1)
        v = random.randint(0, nodes-1)
        if u != v:
            g.add_edge(u, v)
    return g

def create_random_dense(nodes):
    # random dense graph (d=0.5)
    g = Graph()
    max_edges = nodes * (nodes - 1) // 2
    target_edges = int(max_edges * 0.5)
    
    edges_added = 0
    attempts = 0
    while edges_added < target_edges and attempts < target_edges * 3:
        u = random.randint(0, nodes-1)
        v = random.randint(0, nodes-1)
        if u != v and v not in g.graph.get(u, []):
            g.add_edge(u, v)
            edges_added += 1
        attempts += 1
    return g

def measure_performance():
    # test different graph sizes and types
    sizes = [10, 20, 50, 100, 200]
    
    # store results for each graph type
    results = {
        'Chain Graph': {'dfs': [], 'bfs': []},
        'Binary Tree': {'dfs': [], 'bfs': []},
        'Cyclic Graph': {'dfs': [], 'bfs': []},
        'Random Sparse (e=2n)': {'dfs': [], 'bfs': []},
        'Random Dense (d=0.5)': {'dfs': [], 'bfs': []}
    }
    
    print("testing performance on different graph types...")
    
    for size in sizes:
        print(f"testing size {size}")
        
        # create different graph types
        graphs = {
            'Chain Graph': create_chain_graph(size),
            'Binary Tree': create_binary_tree(size),
            'Cyclic Graph': create_cyclic_graph(size),
            'Random Sparse (e=2n)': create_random_sparse(size),
            'Random Dense (d=0.5)': create_random_dense(size)
        }
        
        start = 0
        target = size - 1
        
        for graph_type, g in graphs.items():
            # measure dfs
            start_time = time.time()
            ops_dfs, found_dfs = g.dfs(start, target)
            dfs_time = time.time() - start_time
            
            # measure bfs  
            start_time = time.time()
            ops_bfs, found_bfs = g.bfs(start, target)
            bfs_time = time.time() - start_time
            
            results[graph_type]['dfs'].append(dfs_time * 1000)
            results[graph_type]['bfs'].append(bfs_time * 1000)
            
            print(f"  {graph_type}: dfs {ops_dfs}ops/{dfs_time*1000:.2f}ms, bfs {ops_bfs}ops/{bfs_time*1000:.2f}ms")
    
    return sizes, results

def plot_results(sizes, results):
    # make 2 plots - one for dfs, one for bfs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # colors and markers for each graph type
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    graph_types = ['Chain Graph', 'Binary Tree', 'Cyclic Graph', 'Random Sparse (e=2n)', 'Random Dense (d=0.5)']
    
    # dfs plot
    for i, graph_type in enumerate(graph_types):
        ax1.plot(sizes, results[graph_type]['dfs'], 
                color=colors[i], marker=markers[i], label=graph_type, linewidth=2)
    
    ax1.set_xlabel('Graph Size (nodes)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('DFS Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # bfs plot
    for i, graph_type in enumerate(graph_types):
        ax2.plot(sizes, results[graph_type]['bfs'], 
                color=colors[i], marker=markers[i], label=graph_type, linewidth=2)
    
    ax2.set_xlabel('Graph Size (nodes)')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('BFS Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # run the analysis
    print("starting dfs vs bfs analysis...")
    
    # simple test first
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(0, 3)
    
    print("\nsimple test:")
    dfs_result = g.dfs(0, 3)
    bfs_result = g.bfs(0, 3)
    print(f"dfs: {dfs_result}")
    print(f"bfs: {bfs_result}")
    
    # performance analysis on different graph types
    sizes, results = measure_performance()
    
    # show results in 2 graphs
    plot_results(sizes, results)
    
    print("\nanalysis complete!")