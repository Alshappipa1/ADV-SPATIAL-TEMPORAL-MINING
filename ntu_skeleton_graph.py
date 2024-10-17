import numpy as np
import torch

# We have 25 as # of joints 
num_node = 25  

# Self-connections (each joint is connected to itself) as ST-GCN
self_link = [(i, i) for i in range(num_node)]

# The skeleton structure for NTU RGB+D dataset and how they connected to each other the same as ST-GCN
inward_ori_index = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)
]

# Create inward and outward edges
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


# generate the adjacency matrix
def get_adjacency_matrix():
    A = np.zeros((num_node, num_node))
    for i, j in neighbor + self_link:
        A[i, j] = 1
    A += np.eye(num_node)  # here is self-connections 
    return A

# normalize the adjacency matrix
def normalize_adjacency_matrix(A):
    D = np.diag(np.sum(A, axis=1))  # Degree matrix
    D_inv = np.linalg.inv(D)        # Inverse degree matrix
    return np.dot(D_inv, A)

# Create and normalize the adjacency matrix
def get_normalized_adjacency_matrix():
    A = get_adjacency_matrix()
    A_norm = normalize_adjacency_matrix(A)
    return torch.tensor(A_norm, dtype=torch.float32)

