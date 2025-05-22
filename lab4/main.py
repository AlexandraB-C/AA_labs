import numpy as np
import matplotlib.pyplot as plt
import time
import random
from collections import defaultdict
import heapq

class GraphGen:
    """gen graphs for testing"""
    
    @staticmethod
    def sparse_graph(n, ratio=0.3):
        """sparse graph w/ limited edges"""
        graph = defaultdict(list)
        matrix = [[float('inf')] * n for _ in range(n)]
        
        # diag = 0
        for i in range(n):
            matrix[i][i] = 0
        
        # add edges by ratio
        max_e = n * (n - 1) // 2
        num_e = int(max_e * ratio)
        
        added = 0
        while added < num_e:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            
            if u != v and matrix[u][v] == float('inf'):
                w = random.randint(1, 10)
                graph[u].append((v, w))
                matrix[u][v] = w
                matrix[v][u] = w  # undirected
                added += 1
        
        return graph, matrix
    
    @staticmethod
    def dense_graph(n, ratio=0.8):
        """dense graph w/ many edges"""
        graph = defaultdict(list)
        matrix = [[float('inf')] * n for _ in range(n)]
        
        # diag = 0
        for i in range(n):
            matrix[i][i] = 0
        
        # add edges by ratio
        max_e = n * (n - 1) // 2
        num_e = int(max_e * ratio)
        
        added = 0
        while added < num_e:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            
            if u != v and matrix[u][v] == float('inf'):
                w = random.randint(1, 10)
                graph[u].append((v, w))
                matrix[u][v] = w
                matrix[v][u] = w  # undirected
                added += 1
        
        return graph, matrix

class Dijkstra:
    """dijkstra shortest path impl"""
    
    def __init__(self, graph):
        self.graph = graph
    
    def shortest_path(self, start):
        """find shortest from start to all nodes"""
        dist = defaultdict(lambda: float('inf'))
        dist[start] = 0
        visited = set()
        pq = [(0, start)]  # priority queue
        
        while pq:
            curr_dist, curr = heapq.heappop(pq)
            
            if curr in visited:
                continue
            
            visited.add(curr)
            
            # check neighbors
            for neighbor, weight in self.graph[curr]:
                new_dist = curr_dist + weight
                
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
        
        return dict(dist)

class FloydWarshall:
    """floyd-warshall all-pairs shortest path"""
    
    def __init__(self, matrix):
        self.matrix = [row[:] for row in matrix]  # deep copy
        self.n = len(matrix)
    
    def all_shortest_paths(self):
        """find shortest between all pairs - dp approach"""
        # dp: try each intermediate vertex k
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    # if path through k is shorter
                    if (self.matrix[i][k] != float('inf') and 
                        self.matrix[k][j] != float('inf')):
                        self.matrix[i][j] = min(
                            self.matrix[i][j],
                            self.matrix[i][k] + self.matrix[k][j]
                        )
        
        return self.matrix

class Benchmark:
    """perf analysis & comparison"""
    
    def __init__(self):
        self.results = {
            'dij_sparse': [],
            'dij_dense': [],
            'fw_sparse': [],
            'fw_dense': [],
            'nodes': []
        }
    
    def run_tests(self, node_sizes):
        """run perf tests on diff graph sizes"""
        
        for n in node_sizes:
            print(f"testing {n} nodes...")
            
            # sparse tests
            sparse_g, sparse_m = GraphGen.sparse_graph(n)
            
            # dijkstra sparse
            start = time.time()
            dij = Dijkstra(sparse_g)
            dij.shortest_path(0)
            dij_sparse_t = time.time() - start
            
            # floyd-warshall sparse
            start = time.time()
            fw = FloydWarshall(sparse_m)
            fw.all_shortest_paths()
            fw_sparse_t = time.time() - start
            
            # dense tests
            dense_g, dense_m = GraphGen.dense_graph(n)
            
            # dijkstra dense
            start = time.time()
            dij = Dijkstra(dense_g)
            dij.shortest_path(0)
            dij_dense_t = time.time() - start
            
            # floyd-warshall dense
            start = time.time()
            fw = FloydWarshall(dense_m)
            fw.all_shortest_paths()
            fw_dense_t = time.time() - start
            
            # save results
            self.results['dij_sparse'].append(dij_sparse_t)
            self.results['dij_dense'].append(dij_dense_t)
            self.results['fw_sparse'].append(fw_sparse_t)
            self.results['fw_dense'].append(fw_dense_t)
            self.results['nodes'].append(n)
    
    def make_graphs(self):
        """create perf comparison graphs"""
        
        plt.figure(figsize=(16, 12))
        
        # graph 1: dijkstra comparison
        plt.subplot(2, 3, 1)
        plt.plot(self.results['nodes'], self.results['dij_sparse'], 
                'b-o', label='sparse', linewidth=2, markersize=6)
        plt.plot(self.results['nodes'], self.results['dij_dense'], 
                'r-s', label='dense', linewidth=2, markersize=6)
        plt.xlabel('nodes')
        plt.ylabel('time (sec)')
        plt.title('dijkstra performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # graph 2: floyd-warshall comparison
        plt.subplot(2, 3, 2)
        plt.plot(self.results['nodes'], self.results['fw_sparse'], 
                'g-^', label='sparse', linewidth=2, markersize=6)
        plt.plot(self.results['nodes'], self.results['fw_dense'], 
                'm-d', label='dense', linewidth=2, markersize=6)
        plt.xlabel('nodes')
        plt.ylabel('time (sec)')
        plt.title('floyd-warshall performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # graph 3: sparse comparison
        plt.subplot(2, 3, 3)
        plt.plot(self.results['nodes'], self.results['dij_sparse'], 
                'b-o', label='dijkstra', linewidth=2, markersize=6)
        plt.plot(self.results['nodes'], self.results['fw_sparse'], 
                'g-^', label='floyd-warshall', linewidth=2, markersize=6)
        plt.xlabel('nodes')
        plt.ylabel('time (sec)')
        plt.title('sparse graphs comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # graph 4: dense comparison
        plt.subplot(2, 3, 4)
        plt.plot(self.results['nodes'], self.results['dij_dense'], 
                'r-s', label='dijkstra', linewidth=2, markersize=6)
        plt.plot(self.results['nodes'], self.results['fw_dense'], 
                'm-d', label='floyd-warshall', linewidth=2, markersize=6)
        plt.xlabel('nodes')
        plt.ylabel('time (sec)')
        plt.title('dense graphs comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # graph 5: time complexity visualization
        plt.subplot(2, 3, 5)
        nodes = np.array(self.results['nodes'])
        # theoretical complexity curves
        plt.plot(nodes, (nodes * np.log(nodes)) * 0.001, 'b--', 
                label='o(n log n)', alpha=0.7, linewidth=2)
        plt.plot(nodes, (nodes ** 3) * 0.000001, 'r--', 
                label='o(n³)', alpha=0.7, linewidth=2)
        plt.xlabel('nodes')
        plt.ylabel('normalized time')
        plt.title('theoretical complexity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # graph 6: efficiency ratio
        plt.subplot(2, 3, 6)
        ratio_sparse = [fw/dij for fw, dij in zip(self.results['fw_sparse'], self.results['dij_sparse'])]
        ratio_dense = [fw/dij for fw, dij in zip(self.results['fw_dense'], self.results['dij_dense'])]
        plt.plot(self.results['nodes'], ratio_sparse, 'g-o', 
                label='sparse ratio', linewidth=2, markersize=6)
        plt.plot(self.results['nodes'], ratio_dense, 'm-s', 
                label='dense ratio', linewidth=2, markersize=6)
        plt.xlabel('nodes')
        plt.ylabel('fw/dijkstra ratio')
        plt.title('efficiency ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('perf_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """print detailed analysis"""
        print("\n" + "="*70)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"{'nodes':<8} {'dij_sparse':<12} {'dij_dense':<12} {'fw_sparse':<12} {'fw_dense':<12}")
        print("-" * 70)
        
        for i, n in enumerate(self.results['nodes']):
            print(f"{n:<8} {self.results['dij_sparse'][i]:<12.4f} "
                  f"{self.results['dij_dense'][i]:<12.4f} "
                  f"{self.results['fw_sparse'][i]:<12.4f} "
                  f"{self.results['fw_dense'][i]:<12.4f}")
        
        # calc avg improvement ratios
        avg_sparse_ratio = sum(fw/dij for fw, dij in 
                              zip(self.results['fw_sparse'], self.results['dij_sparse'])) / len(self.results['nodes'])
        avg_dense_ratio = sum(fw/dij for fw, dij in 
                             zip(self.results['fw_dense'], self.results['dij_dense'])) / len(self.results['nodes'])
        
        print(f"\navg fw/dijkstra ratio - sparse: {avg_sparse_ratio:.2f}x")
        print(f"avg fw/dijkstra ratio - dense: {avg_dense_ratio:.2f}x")

def main():
    """main analysis runner"""
    print("dynamic programming algorithms analysis")
    print("="*50)
    
    # setup
    bench = Benchmark()
    
    # test sizes
    sizes = [10, 15, 20, 25, 30, 35, 40]
    
    # run tests
    bench.run_tests(sizes)
    
    # create visuals
    bench.make_graphs()
    
    # show summary
    bench.print_summary()
    
    print("\nanalysis done! check 'perf_analysis.png'")

if __name__ == "__main__":
    main()