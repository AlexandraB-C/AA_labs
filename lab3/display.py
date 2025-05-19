import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def quicksort_visualize(arr, frames, highlight_frames):
    def _quicksort(arr, start, end, depth=0):
        if start >= end:
            return

        pivot = arr[start]
        i = start + 1
        j = end

        while True:
            while i <= j and arr[i] <= pivot:
                i += 1
            while i <= j and arr[j] > pivot:
                j -= 1
            
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                frames.append(arr.copy())
                highlight_frames.append([start, i, j])
            else:
                break
        
        # Move pivot to its final position
        arr[start], arr[j] = arr[j], arr[start]
        frames.append(arr.copy())
        highlight_frames.append([start, j])
        
        # Recursively sort the sub-arrays
        _quicksort(arr, start, j - 1, depth+1)
        _quicksort(arr, j + 1, end, depth+1)
    
    _quicksort(arr, 0, len(arr) - 1)
    return arr

def mergesort_visualize(arr, frames, highlight_frames):
    def _mergesort(arr, start, end):
        if end - start <= 1:
            return
        
        mid = (start + end) // 2
        _mergesort(arr, start, mid)
        _mergesort(arr, mid, end)
        
        left = arr[start:mid].copy()
        right = arr[mid:end].copy()
        
        i = j = 0
        k = start
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
            frames.append(arr.copy())
            highlight_frames.append([k-1])
        
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
            frames.append(arr.copy())
            highlight_frames.append([k-1])
        
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1
            frames.append(arr.copy())
            highlight_frames.append([k-1])
    
    _mergesort(arr, 0, len(arr))
    return arr

def heapsort_visualize(arr, frames, highlight_frames):
    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            frames.append(arr.copy())
            highlight_frames.append([i, largest])
            heapify(arr, n, largest)
    
    n = len(arr)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        frames.append(arr.copy())
        highlight_frames.append([0, i])
        heapify(arr, i, 0)
    
    return arr

def gnome_sort_visualize(arr, frames, highlight_frames):
    n = len(arr)
    index = 0
    while index < n:
        if index == 0:
            index += 1
        
        if index < n and arr[index] >= arr[index - 1]:
            index += 1
        else:
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            frames.append(arr.copy())
            highlight_frames.append([index, index-1])
            index -= 1
    
    return arr

def generate_test_arrays(size=30):
    arrays = {
        "random": [73, -12, 456.5, 89, -234, 5.25, 678, -34.7, 901, 23, -567, 8.9, 123, 45.6, 789, -15,
               321.3, 67, -890, 2, 444, -98.2, 765, 31, -222, 543.8, 999, 17.4, 88, -654, 111.1,
               333, -77, 987.5, 44, 555, -9, 876, 66.6, -432, 11, 777.7, -22, 666, 99.9, -345],
    
        "nearly_sorted": [-1, 3.2, -2, 5, 4.5, -7, 6, 9.1, -8, 11, 10.8, -13, 12, 15.5, -14, 17,
                     16.3, -19, 18, 21.7, -20, 23, 22.9, -25, 24, 27.4, -26, 29, 28.6, -31, 30,
                     33.3, -32, 35, 34.2, -37, 36, 39.9, -38, 41, 40.1, -43, 42, 45.5, -44, 47],
    
        "reverse": [999, 987.5, 965, -943, 921.3, 899, -877, 855.7, 833, -811, 789.9, 767, -745, 723.2,
                701, -679, 657.8, 635, -613, 591.4, 569, -547, 525.6, 503, -481, 459.1, 437,
                -415, 393.3, 371, -349, 327.7, 305, -283, 261.9, 239, -217, 195.5, 173, -151],
    
        "duplicates": [45, 45, -123, -123, 7.5, 7.5, -89, -89, 456.2, 456.2, 23, 23, -678, -678,
                   12.9, 12.9, 333, 333, -9, -9, 555.5, 555.5, 77, 77, -888, -888, 34.4, 34.4,
                   222, 222, -654, -654, 11.1, 11.1, 987, 987, -66, -66, 432.3, 432.3, 99]
    }
    return arrays

def visualize_sorting(sort_func, algo_name):
    test_arrays = generate_test_arrays()
    
    # Create a figure with subplots for each array type
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    axs = axs.flatten()
    plt.tight_layout(pad=1.0)
    fig.suptitle(f"{algo_name} Visualization", fontsize=14)
    
    # Store all frames and highlights for each array type
    all_frames = []
    all_highlights = []
    array_types = []
    
    for i, (arr_type, arr) in enumerate(test_arrays.items()):
        frames = [arr.copy()]
        highlight_frames = [[]]
        
        arr_copy = arr.copy()
        sort_func(arr_copy, frames, highlight_frames)
        
        all_frames.append(frames)
        all_highlights.append(highlight_frames)
        array_types.append(arr_type)
    
    # Find the maximum number of frames
    max_frames = max(len(frames) for frames in all_frames)
    
    def update(frame_num):
        for i, (frames, highlights, arr_type) in enumerate(zip(all_frames, all_highlights, array_types)):
            ax = axs[i]
            ax.clear()
            
            # Get the current frame or the last frame if we've gone past the end
            current_frame = min(frame_num, len(frames) - 1)
            frame = frames[current_frame]
            
            # Get the current highlight or empty list if we've gone past the end
            highlight = highlights[current_frame] if current_frame < len(highlights) else []
            
            # Create the bar chart
            bars = ax.bar(range(len(frame)), frame, color='skyblue')
            
            # Highlight elements being compared/swapped
            for idx in highlight:
                if 0 <= idx < len(frame):
                    bars[idx].set_color('red')
            
            ax.set_title(f"{arr_type} Array (Frame {current_frame+1}/{len(frames)})")
            ax.set_xlim(-1, len(frame))
            
            # Calculate min and max values across all frames for this array type
            all_values = [val for frame in frames for val in frame]
            y_min = min(all_values) - 10
            y_max = max(all_values) + 10
            ax.set_ylim(y_min, y_max)
        
        return [bar for ax in axs for bar in ax.containers]
    
    ani = animation.FuncAnimation(
        fig, update, frames=max_frames, 
        interval=50,
        blit=False, repeat=False
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def main():
    print("\nSorting Algorithm Visualization")
    print("1. QuickSort")
    print("2. MergeSort")
    print("3. HeapSort")
    print("4. Gnome Sort")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    algorithms = {
        "1": ("QuickSort", quicksort_visualize),
        "2": ("MergeSort", mergesort_visualize),
        "3": ("HeapSort", heapsort_visualize),
        "4": ("Gnome Sort", gnome_sort_visualize)
    }
    
    if choice in algorithms:
        algo_name, sort_func = algorithms[choice]
        visualize_sorting(sort_func, algo_name)
    else:
        print("Invalid choice. Please run the program again.")

if __name__ == "__main__":
    main()