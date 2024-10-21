import numpy as np
import os

# Read and load skeleton data/ just random values for now: this part need more works
def read_skeleton(file_path):
    with open(file_path, 'r') as f:
        skeleton_data = f.readlines()
    return np.random.rand(3, 50, 25, 2) #random values with skeleton_data shape for testing


# Load all skeleton files in the specified directory: need more works
def load_data(data_dir):
    skeleton_data = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.skeleton'):
            file_path = os.path.join(data_dir, file_name)
            skeleton = read_skeleton(file_path)
            skeleton_data.append(skeleton)
    
    return np.array(skeleton_data)

