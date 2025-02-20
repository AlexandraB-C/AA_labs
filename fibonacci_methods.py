import time
import matplotlib.pyplot as plt
from functools import lru_cache
import numpy as np
import psutil
import math
import tracemalloc

class FibonacciAnalyzer:
    def __init__(self):
        # series1 (for recursive methods)
        self.series1 = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37]
        # series2 (for non-recursive methods)
        self.series2 = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000]
    
    # 1. naive recursive method
    def fibonacci_recursive(self, n):
        if n <= 1:
            return n
        return self.fibonacci_recursive(n - 1) + self.fibonacci_recursive(n - 2)
    
    # 2. recursive method with memoization
    @lru_cache(maxsize=None)
    def fibonacci_memoization(self, n):
        if n <= 1:
            return n
        return self.fibonacci_memoization(n - 1) + self.fibonacci_memoization(n - 2)
    
    # 3. dynamic programming method
    def fibonacci_dp(self, n):
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]
    
    # 4. space optimized method
    def fibonacci_space_optimized(self, n):
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    # 5. matrix exponentiation method
    def matrix_power(self, matrix, n):
        if n == 0:
            return np.identity(2, dtype=object)
        if n == 1:
            return matrix
        if n % 2 == 0:
            half = self.matrix_power(matrix, n // 2)
            return np.dot(half, half)
        else:
            return np.dot(matrix, self.matrix_power(matrix, n - 1))
    
    def fibonacci_matrix(self, n):
        if n <= 1:
            return n
        base_matrix = np.array([[1, 1], [1, 0]], dtype=object)
        result_matrix = self.matrix_power(base_matrix, n - 1)
        return result_matrix[0][0]
    
    # 6. Binet's formula
    # direct formula, very fast, not perfect for big numbers
    def fibonacci_binet(self, n):
        phi = (1 + math.sqrt(5)) / 2
        return round((phi**n - (-1/phi)**n) / math.sqrt(5))
    
    # measures time and memory used by a method
    def measure_performance(self, method, series):
        time_taken = []
        space_used = []
        
        for n in series:
            tracemalloc.start()
            start_time = time.time()
            method(n)
            end_time = time.time()
            
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            elapsed_time = end_time - start_time
            time_taken.append(elapsed_time)
            space_used.append(peak_memory / 1024)
            
            print(f"{method.__name__}({n}), {elapsed_time:.6f} s, {peak_memory / 1024:.2f} KB")
        
        return time_taken, space_used
    
    # run tests on all methods
    def run_analysis(self):
        results = {}
        
        print("\nAnalyzing recursive methods:")
        for name, func in [("recursive", self.fibonacci_recursive),
                           ("memoization", self.fibonacci_memoization),
                           ("binet", self.fibonacci_binet)]:
            results[name] = self.measure_performance(func, self.series1)
        
        print("\nAnalyzing non-recursive methods:")
        for name, func in [("dynamic", self.fibonacci_dp),
                           ("space optimized", self.fibonacci_space_optimized),
                           ("matrix exp", self.fibonacci_matrix)]:
            results[name] = self.measure_performance(func, self.series2)
        
        # draw time and memory plots
        for name, (times, memory) in results.items():
            plt.figure()
            ns = self.series1 if name in ["recursive", "memoization", "binet"] else self.series2
            plt.plot(ns, times, marker='o', linestyle='-', label=f"{name} time")
            plt.xlabel("n")
            plt.ylabel("time (s)")
            plt.title(f"Performance of {name} method")
            plt.legend()
            plt.grid()
            plt.show()
            
            plt.figure()
            plt.plot(ns, memory, marker='o', linestyle='-', label=f"{name} memory usage")
            plt.xlabel("n")
            plt.ylabel("memory (KB)")
            plt.title(f"Memory usage of {name} method")
            plt.legend()
            plt.grid()
            plt.show()
        
        # all times together
        plt.figure(figsize=(10, 6))
        for name, (times, _) in results.items():
            ns = self.series1 if name in ["recursive", "memoization", "binet"] else self.series2
            plt.plot(ns, times, marker='o', linestyle='-', label=name)
        plt.xlabel("n")
        plt.ylabel("time (s)")
        plt.title("Performance Comparison of Fibonacci Algorithms")
        plt.legend()
        plt.grid()
        plt.yscale("log")
        plt.show()

if __name__ == "__main__":
    analyzer = FibonacciAnalyzer()
    analyzer.run_analysis()
