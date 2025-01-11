import hnswlib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor
import time
import argparse

dim = 128
k = 10

parser = argparse.ArgumentParser()

parser.add_argument("--num_elements", type=int, required=True)
parser.add_argument("--num_threads", type=int, required=True)

args = parser.parse_args()

num_elements = args.num_elements
num_threads = args.num_threads

np.random.seed(42)
data = np.random.random((num_elements, dim)).astype(np.float32)
query_data = np.random.random((100, dim)).astype(np.float32)  

nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
nn.fit(data)
ground_truth = nn.kneighbors(query_data, return_distance=False)

p = hnswlib.Index(space='l2', dim=dim)
p.init_index(max_elements=num_elements, ef_construction=400, M=32)

print(num_elements)

start_time = time.time()
p.add_items(data, num_threads = num_threads)
end_time = time.time()

cost = end_time - start_time

print(f"Insert {num_elements} elements, total time for {num_threads}-threaded add_items: {cost:.4f} seconds")

print(f"insert qps: {num_elements / cost:.2f}")

p.set_ef(200) 
start = timer()
labels, _ = p.knn_query(query_data, k=k)
end = timer()

# print(f"knn_query took {end - start:.4f} seconds")

recall_list = []
for i in range(len(query_data)):
    recall = len(set(labels[i]).intersection(set(ground_truth[i]))) / k
    recall_list.append(recall)

avg_recall = np.mean(recall_list)
print(f"Average Recall: {avg_recall:.4f}")

p.print_stat()
