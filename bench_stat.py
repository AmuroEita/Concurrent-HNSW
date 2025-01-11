import subprocess
import re
import matplotlib.pyplot as plt
import os
import numpy as np

# Input parameters
num_threads_list = [4, 8, 16, 32, 48, 64]
num_elements = 100000

# Create output directory
output_dir = "plots-10k"
os.makedirs(output_dir, exist_ok=True)

# Function to execute a command
def run_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error message: {e.stderr}")
        return None

# Regex patterns for extracting data
insert_time_pattern = r"total time for .*?add_items:\s*([\d\.]+)\s*seconds"
search_layer_contention_time_pattern = r"Total lock contention of search layer time \(in ns\):\s*(\d+)"
search_layer0_contention_time_pattern = r"Total lock contention of search layer0 time \(in ns\):\s*(\d+)"
link_contention_time_pattern = r"Total lock contention of link time \(in ns\):\s*(\d+)"

# Data storage
search_layer_lock_time_total_list = []
search_layer0_lock_time_total_list = []
link_lock_time_total_list = []
insert_time_list = []
insert_time_per_thread_list = []

# Run the benchmark and extract data
for num_threads in num_threads_list:
    cmd = f"python3 bench.py --num_elements {num_elements} --num_threads {num_threads}"
    output = run_command(cmd)
    if output:
        # Extract Insert Time
        insert_time_match = re.search(insert_time_pattern, output)
        insert_time = float(insert_time_match.group(1)) * 1000 if insert_time_match else 0  # Convert to ms
        insert_time_list.append(insert_time)
        insert_time_per_thread_list.append(insert_time / num_threads)

        # Extract lock contention times
        search_layer_contention_time_match = re.search(search_layer_contention_time_pattern, output)
        search_layer_lock_time = int(search_layer_contention_time_match.group(1)) if search_layer_contention_time_match else 0
        search_layer_lock_time_total_list.append(search_layer_lock_time)

        search_layer0_contention_time_match = re.search(search_layer0_contention_time_pattern, output)
        search_layer0_lock_time = int(search_layer0_contention_time_match.group(1)) if search_layer0_contention_time_match else 0
        search_layer0_lock_time_total_list.append(search_layer0_lock_time)

        link_contention_time_match = re.search(link_contention_time_pattern, output)
        link_lock_time = int(link_contention_time_match.group(1)) if link_contention_time_match else 0
        link_lock_time_total_list.append(link_lock_time)

# Convert data to NumPy arrays
total_lock_time_array = np.array([search_layer_lock_time_total_list, 
                                  search_layer0_lock_time_total_list, 
                                  link_lock_time_total_list])
labels = ["Search Layer", "Search Layer0", "Link"]

# Total lock contention time and total insert time
plt.figure(figsize=(12, 6))
plt.bar(num_threads_list, total_lock_time_array[0] / 1e6, label=labels[0], color='green', alpha=0.7)
plt.bar(num_threads_list, total_lock_time_array[1] / 1e6, bottom=total_lock_time_array[0] / 1e6, label=labels[1], color='orange', alpha=0.7)
plt.bar(num_threads_list, total_lock_time_array[2] / 1e6, bottom=(total_lock_time_array[0] + total_lock_time_array[1]) / 1e6, label=labels[2], color='blue', alpha=0.7)

# Add total insert time
plt.plot(num_threads_list, np.array(insert_time_list), marker='o', label='Insert Time Total (ms)', color='red', linewidth=2)

plt.xlabel('Threads')
plt.ylabel('Time (ms)')
plt.title('Total Lock Time and Insert Time Total vs Threads')
plt.legend()
plt.grid(axis='y')
total_lock_time_path = os.path.join(output_dir, "LOCK total_lock_time_and_insert_time_vs_threads.png")
plt.savefig(total_lock_time_path)
plt.close()

# Per-thread lock contention time and per-thread insert time
plt.figure(figsize=(12, 6))
per_thread_lock_time_array = total_lock_time_array / np.array(num_threads_list)[None, :]

plt.bar(num_threads_list, per_thread_lock_time_array[0] / 1e6, label=labels[0], color='green', alpha=0.7)
plt.bar(num_threads_list, per_thread_lock_time_array[1] / 1e6, bottom=per_thread_lock_time_array[0] / 1e6, label=labels[1], color='orange', alpha=0.7)
plt.bar(num_threads_list, per_thread_lock_time_array[2] / 1e6, bottom=(per_thread_lock_time_array[0] + per_thread_lock_time_array[1]) / 1e6, label=labels[2], color='blue', alpha=0.7)

# Add per-thread insert time
plt.plot(num_threads_list, np.array(insert_time_per_thread_list), marker='o', label='Insert Time Per Thread (ms)', color='red', linewidth=2)

plt.xlabel('Threads')
plt.ylabel('Per Thread Time (ms)')
plt.title('Per Thread Lock Time and Insert Time Per Thread vs Threads')
plt.legend()
plt.grid(axis='y')
per_thread_lock_time_path = os.path.join(output_dir, "LOCK per_thread_lock_time_and_insert_time_vs_threads.png")
plt.savefig(per_thread_lock_time_path)
plt.close()

print(f"Total lock time and insert time plot saved as: {total_lock_time_path}")
print(f"Per thread lock time and insert time plot saved as: {per_thread_lock_time_path}")
