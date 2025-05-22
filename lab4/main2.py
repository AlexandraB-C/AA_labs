import numpy as np
import matplotlib.pyplot as plt
import time
import random
from collections import defaultdict
import heapq

class GraphGen:
    """Generate graphs for testing"""
    
    @staticmethod
    def sparse_graph(n, ratio=0.3):
        """Generate sparse graph with limited edges"""
        graph = defaultdict(list)
        matrix = [[float('inf')] * n for _ in range(n)]
        
        # Diagonal = 0
        for i in range(n):
            matrix[i][i] = 0
        
        # Ensure graph is connected first
        nodes = list(range(n))
        random.shuffle(nodes)
        
        # Create minimum spanning tree to ensure connectivity
        for i in range(n - 1):
            u, v = nodes[i], nodes[i + 1]
            w = random.randint(1, 10)
            graph[u].append((v, w))
            graph[v].append((u, w))
            matrix[u][v] = w
            matrix[v][u] = w
        
        # Add additional edges based on ratio
        max_e = n * (n - 1) // 2
        target_e = int(max_e * ratio)
        current_e = n - 1  # edges from MST
        
        attempts = 0
        max_attempts = target_e * 3  # Avoid infinite loops
        
        while current_e < target_e and attempts < max_attempts:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            attempts += 1
            
            if u != v and matrix[u][v] == float('inf'):
                w = random.randint(1, 10)
                graph[u].append((v, w))
                graph[v].append((u, w))
                matrix[u][v] = w
                matrix[v][u] = w
                current_e += 1
        
        return graph, matrix
    
    @staticmethod
    def dense_graph(n, ratio=0.8):
        """Generate dense graph with many edges"""
        return GraphGen.sparse_graph(n, ratio)

class Dijkstra:
    """Improved Dijkstra shortest path implementation"""
    
    def __init__(self, graph=None, matrix=None):
        if graph is not None:
            self.graph = graph
        elif matrix is not None:
            # Convert matrix to adjacency list
            self.graph = defaultdict(list)
            n = len(matrix)
            for i in range(n):
                for j in range(n):
                    if i != j and matrix[i][j] != float('inf'):
                        self.graph[i].append((j, matrix[i][j]))
        else:
            raise ValueError("Either graph or matrix must be provided")
    
    def shortest_path_single(self, start, end=None):
        """Find shortest path from start to end (or all nodes if end=None)"""
        dist = defaultdict(lambda: float('inf'))
        dist[start] = 0
        visited = set()
        pq = [(0, start)]
        parent = {}
        
        while pq:
            curr_dist, curr = heapq.heappop(pq)
            
            if curr in visited:
                continue
            
            visited.add(curr)
            
            # If we only need path to specific end node
            if end is not None and curr == end:
                break
            
            # Check neighbors
            for neighbor, weight in self.graph[curr]:
                if neighbor not in visited:
                    new_dist = curr_dist + weight
                    
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        parent[neighbor] = curr
                        heapq.heappush(pq, (new_dist, neighbor))
        
        return dict(dist), parent
    
    def all_pairs_shortest_paths(self):
        """Find shortest paths between all pairs using repeated Dijkstra"""
        # Get all nodes
        all_nodes = set()
        for node in self.graph:
            all_nodes.add(node)
            for neighbor, _ in self.graph[node]:
                all_nodes.add(neighbor)
        
        all_nodes = sorted(list(all_nodes))
        n = len(all_nodes)
        
        # Initialize result matrix
        result = [[float('inf')] * n for _ in range(n)]
        
        # Set diagonal to 0
        for i in range(n):
            result[i][i] = 0
        
        # Run Dijkstra from each node
        for i, start_node in enumerate(all_nodes):
            distances, _ = self.shortest_path_single(start_node)
            for j, end_node in enumerate(all_nodes):
                if end_node in distances:
                    result[i][j] = distances[end_node]
        
        return result
    
    def get_path(self, parent, start, end):
        """Reconstruct path from parent dictionary"""
        if end not in parent and end != start:
            return None
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = parent.get(current)
        
        path.reverse()
        return path if path[0] == start else None

class FloydWarshall:
    """Floyd-Warshall all-pairs shortest path"""
    
    def __init__(self, matrix):
        self.matrix = [row[:] for row in matrix]  # Deep copy
        self.n = len(matrix)
    
    def all_shortest_paths(self):
        """Find shortest paths between all pairs using dynamic programming"""
        # DP: try each intermediate vertex k
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    # If path through k is shorter
                    if (self.matrix[i][k] != float('inf') and 
                        self.matrix[k][j] != float('inf')):
                        self.matrix[i][j] = min(
                            self.matrix[i][j],
                            self.matrix[i][k] + self.matrix[k][j]
                        )
        
        return self.matrix

class Benchmark:
    """Performance analysis & comparison"""
    
    def __init__(self):
        self.results = {
            'dij_single_sparse': [],
            'dij_single_dense': [],
            'dij_all_sparse': [],
            'dij_all_dense': [],
            'fw_sparse': [],
            'fw_dense': [],
            'nodes': []
        }
    
    def run_tests(self, node_sizes):
        """Run performance tests on different graph sizes"""
        
        for n in node_sizes:
            print(f"Testing {n} nodes...")
            
            # Sparse tests
            sparse_g, sparse_m = GraphGen.sparse_graph(n)
            
            # Dijkstra single source (sparse)
            start = time.time()
            dij = Dijkstra(graph=sparse_g)
            dij.shortest_path_single(0)
            dij_single_sparse_t = time.time() - start
            
            # Dijkstra all pairs (sparse)
            start = time.time()
            dij = Dijkstra(graph=sparse_g)
            dij.all_pairs_shortest_paths()
            dij_all_sparse_t = time.time() - start
            
            # Floyd-Warshall sparse
            start = time.time()
            fw = FloydWarshall(sparse_m)
            fw.all_shortest_paths()
            fw_sparse_t = time.time() - start
            
            # Dense tests
            dense_g, dense_m = GraphGen.dense_graph(n)
            
            # Dijkstra single source (dense)
            start = time.time()
            dij = Dijkstra(graph=dense_g)
            dij.shortest_path_single(0)
            dij_single_dense_t = time.time() - start
            
            # Dijkstra all pairs (dense)
            start = time.time()
            dij = Dijkstra(graph=dense_g)
            dij.all_pairs_shortest_paths()
            dij_all_dense_t = time.time() - start
            
            # Floyd-Warshall dense
            start = time.time()
            fw = FloydWarshall(dense_m)
            fw.all_shortest_paths()
            fw_dense_t = time.time() - start
            
            # Save results
            self.results['dij_single_sparse'].append(dij_single_sparse_t)
            self.results['dij_single_dense'].append(dij_single_dense_t)
            self.results['dij_all_sparse'].append(dij_all_sparse_t)
            self.results['dij_all_dense'].append(dij_all_dense_t)
            self.results['fw_sparse'].append(fw_sparse_t)
            self.results['fw_dense'].append(fw_dense_t)
            self.results['nodes'].append(n)
    
    def make_graphs(self):
        """Create performance comparison graphs"""
        
        plt.figure(figsize=(20, 12))
        
        # Graph 1: Single-source Dijkstra comparison
        plt.subplot(2, 4, 1)
        plt.plot(self.results['nodes'], self.results['dij_single_sparse'], 
                'b-o', label='sparse', linewidth=2, markersize=6)
        plt.plot(self.results['nodes'], self.results['dij_single_dense'], 
                'r-s', label='dense', linewidth=2, markersize=6)
        plt.xlabel('Nodes')
        plt.ylabel('Time (sec)')
        plt.title('Dijkstra Single-Source Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graph 2: All-pairs comparison (fair comparison)
        plt.subplot(2, 4, 2)
        plt.plot(self.results['nodes'], self.results['dij_all_sparse'], 
                'b-o', label='dijkstra sparse', linewidth=2, markersize=6)
        plt.plot(self.results['nodes'], self.results['fw_sparse'], 
                'g-^', label='floyd-warshall sparse', linewidth=2, markersize=6)
        plt.xlabel('Nodes')
        plt.ylabel('Time (sec)')
        plt.title('All-Pairs Shortest Path (Sparse)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Graph 3: All-pairs comparison (dense)
        plt.subplot(2, 4, 3)
        plt.plot(self.results['nodes'], self.results['dij_all_dense'], 
                'r-s', label='dijkstra dense', linewidth=2, markersize=6)
        plt.plot(self.results['nodes'], self.results['fw_dense'], 
                'm-d', label='floyd-warshall dense', linewidth=2, markersize=6)
        plt.xlabel('Nodes')
        plt.ylabel('Time (sec)')
        plt.title('All-Pairs Shortest Path (Dense)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Graph 4: Floyd-Warshall comparison
        plt.subplot(2, 4, 4)
        plt.plot(self.results['nodes'], self.results['fw_sparse'], 
                'g-^', label='sparse', linewidth=2, markersize=6)
        plt.plot(self.results['nodes'], self.results['fw_dense'], 
                'm-d', label='dense', linewidth=2, markersize=6)
        plt.xlabel('Nodes')
        plt.ylabel('Time (sec)')
        plt.title('Floyd-Warshall Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graph 5: Time complexity visualization
        plt.subplot(2, 4, 5)
        nodes = np.array(self.results['nodes'])
        # Theoretical complexity curves (normalized)
        n_log_n = nodes * np.log2(nodes)
        n_squared = nodes ** 2
        n_cubed = nodes ** 3
        
        # Normalize to fit on same scale
        scale_nlogn = max(self.results['dij_single_sparse']) / max(n_log_n)
        scale_n2 = max(self.results['dij_all_sparse']) / max(n_squared)
        scale_n3 = max(self.results['fw_sparse']) / max(n_cubed)
        
        plt.plot(nodes, n_log_n * scale_nlogn, 'b--', 
                label='O(V log V) - Single Dijkstra', alpha=0.7, linewidth=2)
        plt.plot(nodes, n_squared * scale_n2, 'r--', 
                label='O(V²) - All-pairs Dijkstra', alpha=0.7, linewidth=2)
        plt.plot(nodes, n_cubed * scale_n3, 'g--', 
                label='O(V³) - Floyd-Warshall', alpha=0.7, linewidth=2)
        plt.xlabel('Nodes')
        plt.ylabel('Normalized Time')
        plt.title('Theoretical Complexity Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Graph 6: Efficiency ratio (all-pairs)
        plt.subplot(2, 4, 6)
        ratio_sparse = [fw/dij for fw, dij in zip(self.results['fw_sparse'], self.results['dij_all_sparse'])]
        ratio_dense = [fw/dij for fw, dij in zip(self.results['fw_dense'], self.results['dij_all_dense'])]
        plt.plot(self.results['nodes'], ratio_sparse, 'g-o', 
                label='sparse graphs', linewidth=2, markersize=6)
        plt.plot(self.results['nodes'], ratio_dense, 'm-s', 
                label='dense graphs', linewidth=2, markersize=6)
        plt.xlabel('Nodes')
        plt.ylabel('Floyd-Warshall / Dijkstra Ratio')
        plt.title('Efficiency Ratio (All-Pairs)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graph 7: Memory vs Time tradeoff
        plt.subplot(2, 4, 7)
        nodes = np.array(self.results['nodes'])
        # Approximate memory usage (relative)
        dijkstra_mem = nodes  # O(V) for priority queue
        floyd_mem = nodes ** 2  # O(V²) for matrix
        
        plt.plot(nodes, dijkstra_mem, 'b-', label='Dijkstra Memory', linewidth=2)
        plt.plot(nodes, floyd_mem, 'r-', label='Floyd-Warshall Memory', linewidth=2)
        plt.xlabel('Nodes')
        plt.ylabel('Relative Memory Usage')
        plt.title('Memory Usage Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Graph 8: Performance per operation
        plt.subplot(2, 4, 8)
        nodes = np.array(self.results['nodes'])
        operations = nodes ** 2  # number of pairs
        dij_per_op = np.array(self.results['dij_all_sparse']) / operations
        fw_per_op = np.array(self.results['fw_sparse']) / operations
        
        plt.plot(nodes, dij_per_op * 1000000, 'b-o', 
                label='dijkstra (µs/pair)', linewidth=2, markersize=6)
        plt.plot(nodes, fw_per_op * 1000000, 'g-^', 
                label='floyd-warshall (µs/pair)', linewidth=2, markersize=6)
        plt.xlabel('Nodes')
        plt.ylabel('Time per pair (microseconds)')
        plt.title('Performance per Operation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """Print detailed analysis"""
        print("\n" + "="*90)
        print("COMPREHENSIVE PERFORMANCE ANALYSIS SUMMARY")
        print("="*90)
        
        print(f"{'Nodes':<8} {'Dij-Single':<12} {'Dij-All':<12} {'Floyd-W':<12} {'Speedup':<12} {'Efficiency':<12}")
        print(f"{'':^8} {'Sparse':<12} {'Sparse':<12} {'Sparse':<12} {'FW/Dij':<12} {'µs/pair':<12}")
        print("-" * 90)
        
        for i, n in enumerate(self.results['nodes']):
            speedup = self.results['fw_sparse'][i] / self.results['dij_all_sparse'][i]
            efficiency = (self.results['dij_all_sparse'][i] / (n*n)) * 1000000
            
            print(f"{n:<8} {self.results['dij_single_sparse'][i]:<12.4f} "
                  f"{self.results['dij_all_sparse'][i]:<12.4f} "
                  f"{self.results['fw_sparse'][i]:<12.4f} "
                  f"{speedup:<12.2f} {efficiency:<12.2f}")
        
        print("\nKEY INSIGHTS:")
        print("-" * 50)
        
        # Find crossover point
        crossover = None
        for i in range(len(self.results['nodes'])):
            if self.results['fw_sparse'][i] < self.results['dij_all_sparse'][i]:
                crossover = self.results['nodes'][i]
                break
        
        if crossover:
            print(f"• Floyd-Warshall becomes faster than Dijkstra at ~{crossover} nodes")
        else:
            print("• Dijkstra remains faster than Floyd-Warshall for all tested sizes")
        
        avg_sparse_ratio = sum(fw/dij for fw, dij in 
                              zip(self.results['fw_sparse'], self.results['dij_all_sparse'])) / len(self.results['nodes'])
        
        print(f"• Average Floyd-Warshall/Dijkstra ratio: {avg_sparse_ratio:.2f}x")
        print(f"• Single-source Dijkstra is most efficient for sparse connectivity queries")
        print(f"• Floyd-Warshall has better cache locality and simpler implementation")

def main():
    """Main analysis runner"""
    print("CORRECTED SHORTEST PATH ALGORITHMS ANALYSIS")
    print("="*60)
    
    # Setup
    bench = Benchmark()
    
    # Test sizes (reduced for reasonable runtime)
    sizes = [10, 20, 30, 50, 75, 100]
    
    print(f"Testing with graph sizes: {sizes}")
    print("This may take a few minutes for larger graphs...\n")
    
    # Run tests
    bench.run_tests(sizes)
    
    # Create visuals
    bench.make_graphs()
    
    # Show summary
    bench.print_summary()
    
    print(f"\nAnalysis complete! Check 'comprehensive_analysis.png' for detailed visualizations.")
    
    # Quick demo of corrected functionality
    print("\n" + "="*60)
    print("ALGORITHM CORRECTNESS DEMO")
    print("="*60)
    
    # Create small test graph
    test_graph, test_matrix = GraphGen.sparse_graph(5, 0.6)
    
    print("Test graph adjacency list:")
    for node, edges in test_graph.items():
        print(f"  {node}: {edges}")
    
    # Test Dijkstra
    dij = Dijkstra(graph=test_graph)
    distances, parents = dij.shortest_path_single(0)
    print(f"\nDijkstra from node 0: {distances}")
    
    # Show a path
    if 4 in distances:
        path = dij.get_path(parents, 0, 4)
        print(f"Path from 0 to 4: {path}")
    
    # Test Floyd-Warshall
    fw = FloydWarshall(test_matrix)
    fw_result = fw.all_shortest_paths()
    print(f"\nFloyd-Warshall all-pairs (first row): {fw_result[0]}")

if __name__ == "__main__":
    main()