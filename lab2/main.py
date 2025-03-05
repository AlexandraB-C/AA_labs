import random
import time
import sys

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[-1]
    left = []
    right = []
    for x in arr[:-1]:
        if x <= pivot:
            left.append(x)
        else:
            right.append(x)
    return quicksort(left) + [pivot] + quicksort(right)

def mergesort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    left = mergesort(left)
    right = mergesort(right)
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

TEST_ARRAYS = {
    "random": [73, -12, 456.5, 89, -234, 5.25, 678, -34.7, 901, 23, -567, 8.9, 123, 45.6, 789, -15,
               321.3, 67, -890, 2, 444, -98.2, 765, 31, -222, 543.8, 999, 17.4, 88, -654, 111.1,
               333, -77, 987.5, 44, 555, -9, 876, 66.6, -432, 11, 777.7, -22, 666, 99.9, -345,
               55, -888, 33.3, 234, -7, 654.2, 13, -456, 88, 321, -19.8, 789, 41.5, -567, 3, 912.9],
    
    "nearly_sorted": [-1, 3.2, -2, 5, 4.5, -7, 6, 9.1, -8, 11, 10.8, -13, 12, 15.5, -14, 17,
                     16.3, -19, 18, 21.7, -20, 23, 22.9, -25, 24, 27.4, -26, 29, 28.6, -31, 30,
                     33.3, -32, 35, 34.2, -37, 36, 39.9, -38, 41, 40.1, -43, 42, 45.5, -44, 47,
                     46.7, -49, 48, 51.2, -50, 53, 52.8, -55, 54, 57.6, -56, 59, 58.4, -61, 60,
                     63.3, -62, 64],
    
    "reverse": [999, 987.5, 965, -943, 921.3, 899, -877, 855.7, 833, -811, 789.9, 767, -745, 723.2,
                701, -679, 657.8, 635, -613, 591.4, 569, -547, 525.6, 503, -481, 459.1, 437,
                -415, 393.3, 371, -349, 327.7, 305, -283, 261.9, 239, -217, 195.5, 173, -151,
                129.2, 107, -85, 63.8, 41, -39, 37.6, 35, -33, 31.4, 29, -27, 25.1, 23, -21,
                19.9, 17, -15, 13.3, 11, -9, 7.7, 5, -3, 1.2],
    
    "duplicates": [45, 45, -123, -123, 7.5, 7.5, -89, -89, 456.2, 456.2, 23, 23, -678, -678,
                   12.9, 12.9, 333, 333, -9, -9, 555.5, 555.5, 77, 77, -888, -888, 34.4, 34.4,
                   222, 222, -654, -654, 11.1, 11.1, 987, 987, -66, -66, 432.3, 432.3, 99,
                   99, -345, -345, 15.7, 15.7, 789, 789, -41, -41, 567.8, 567.8, 3, 3, -912,
                   -912, 88.6, 88.6, 321, 321, -19, -19, 44.2, 44.2],
    
    "few_unique": [-5, 17.5, -5, -42, 17.5, 99.9, -5, -42, 17.5, 99.9, -5, 17.5, -42, -5,
                   99.9, 17.5, -5, -42, 17.5, 99.9, -5, 17.5, -42, -5, 99.9, 17.5, -5, -42,
                   17.5, 99.9, -5, 17.5, -42, -5, 99.9, 17.5, -5, -42, 17.5, 99.9, -5, 17.5,
                   -42, -5, 99.9, 17.5, -5, -42, 17.5, 99.9, -5, 17.5, -42, -5, 99.9, 17.5,
                   -5, -42, 17.5, 99.9, -5, 17.5, -42, 3.2]
}

def validate_input(user_input):
    # handle all the weird input cases
    clean_numbers = []
    for item in user_input.replace(',', '.').split():
        try:
            num = float(item)
            clean_numbers.append(num)
        except ValueError:
            print(f"Skipping invalid input: {item}")
    return clean_numbers

def measure_performance(sort_func, arr):
    start_time = time.time()
    sorted_arr = sort_func(arr.copy())
    end_time = time.time()
    memory_used = sys.getsizeof(sorted_arr)
    return {
        "time": (end_time - start_time) * 1000,
        "memory": memory_used,
        "sorted_array": sorted_arr
    }

def print_conclusion(all_stats):
    print("\n=== BATTLE CONCLUSION ===")
    for algo_name, stats in all_stats.items():
        total_time = sum(measures["time"] for measures in stats.values())
        avg_time = total_time / len(stats)
        avg_memory = sum(measures["memory"] for measures in stats.values()) / len(stats)
        
        print(f"\n{algo_name}:")
        print(f"Average Time: {avg_time:.4f} ms")
        print(f"Average Memory: {avg_memory:.2f} bytes")
        
        # find best and worst cases
        times = [(arr_type, measures["time"]) for arr_type, measures in stats.items()]
        best_case = min(times, key=lambda x: x[1])
        worst_case = max(times, key=lambda x: x[1])
        
        print(f"Best performance: {best_case[0]} array ({best_case[1]:.4f} ms)")
        print(f"Worst performance: {worst_case[0]} array ({worst_case[1]:.4f} ms)")

def main():
    print("Welcome  >:3")
    
    print("\nSelect your sorting algorithm:")
    print("1. QuickSort")
    print("2. MergeSort")
    print("3. Run both algorithms")
    print("4. Input custom array")
    
    choice = input("Enter your choice (1-4): ")
    show_sorted = input("\nWanna see the sorted arrays? (y/n): ").lower() == 'y'
    
    all_stats = {}
    
    if choice == '4':
        print("\nEnter numbers:")
        user_input = input()
        custom_array = validate_input(user_input)
        if not custom_array:
            print("No valid numbers provided :c")
            return
        
        print("\nOriginal array:", custom_array)
        stats = measure_performance(quicksort, custom_array)
        print("\nQuickSort Results:")
        print(f"Time taken: {stats['time']:.4f} ms")
        print(f"Memory used: {stats['memory']} bytes")
        if show_sorted:
            print(f"Sorted array: {stats['sorted_array']}")
            
    elif choice in ['1', '2', '3']:
        print("\nOriginal arrays:")
        for name, arr in TEST_ARRAYS.items():
            print(f"{name}: {arr}")
        
        if choice == '1':
            sort_func = quicksort
            print("\n>>> QuickSort Results <<<")
            stats = {name: measure_performance(sort_func, arr) 
                    for name, arr in TEST_ARRAYS.items()}
            all_stats['QuickSort'] = stats
            
        elif choice == '2':
            sort_func = mergesort
            print("\n>>> MergeSort Results <<<")
            stats = {name: measure_performance(sort_func, arr) 
                    for name, arr in TEST_ARRAYS.items()}
            all_stats['MergeSort'] = stats
            
        else:
            print("\n>>> Running Both Algorithms <<<")
            all_stats['QuickSort'] = {name: measure_performance(quicksort, arr) 
                                    for name, arr in TEST_ARRAYS.items()}
            all_stats['MergeSort'] = {name: measure_performance(mergesort, arr) 
                                    for name, arr in TEST_ARRAYS.items()}
        
        for algo_name, stats in all_stats.items():
            print(f"\n{algo_name} Results:")
            for arr_type, measures in stats.items():
                print(f"\n{arr_type} array:")
                print(f"Time taken: {measures['time']:.4f} ms")
                print(f"Memory used: {measures['memory']} bytes")
                if show_sorted:
                    print(f"Sorted array: {measures['sorted_array']}")
        
        print_conclusion(all_stats)

if __name__ == "__main__":
    main()