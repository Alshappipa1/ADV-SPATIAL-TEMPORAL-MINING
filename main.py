import torch
from model import GCNModel
from data import load_skeleton_data
from ntu_skeleton_graph import get_normalized_adjacency_matrix

# the Path to dataset (NTU RGB+D)
data_path = '/home/adel-h/Downloads/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons'

# Load the dataset(test)
skeleton_data = load_skeleton_data(data_path)

# Convert to PyTorch tensor for passing into GCN model for training (for now just for practice)
skeleton_data_tensor = torch.tensor(skeleton_data, dtype=torch.float32)
skeleton_data_tensor = skeleton_data_tensor.mean(dim=-1)  

# Get the normalized adjacency matrix after degree matrix
A = get_normalized_adjacency_matrix()

print(A.shape) #25 joints, size for the adjacency matrix

#------------------------- Below part of the code is just for practice for now. --------------------------------------------------

# Initialize the model
in_channels = 3  # Channels
out_channels = 16  # Number of output channels after GCN
num_classes = 60  # We have 60 action classes in our dataset


#GCN model is started here!
model = GCNModel(in_channels, out_channels, num_classes)


print("Skeleton data tensor shape:", skeleton_data_tensor.shape)  
output = model(skeleton_data_tensor, A)
print("Output shape:", output.shape)


