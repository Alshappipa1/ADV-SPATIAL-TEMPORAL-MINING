import torch
from model import GCNModel
from data import load_data
from ntu_skeleton_graph import get_normalized_adjacency_matrix

# the Path to dataset (NTU RGB+D)
data_path = '/home/adel-h/Downloads/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons'

# Load the dataset(test)
skeleton_data = load_data(data_path)

# Convert to PyTorch tensor for passing into GCN model for training (for now just for practice)
skeleton_d_tensor = torch.tensor(skeleton_data, dtype=torch.float32)
skeleton_d_tensor = skeleton_d_tensor.mean(dim=-1)  

# Get the normalized adjacency matrix after degree matrix(D)
NA = get_normalized_adjacency_matrix()

print(NA.shape) #25 joints, size for the adjacency matrix

#------------------------- Below part of the code is just for practice for now. --------------------------------------------------

# Initialize the model
channels = 3  # Channels
out_channels = 16  # Number of output channels after GCN
num_classes = 60  # We have 60 action classes in our dataset


#GCN model is started here!
model = GCNModel(channels, out_channels, num_classes)


print("Tensor shape:", skeleton_d_tensor.shape)  
output = model(skeleton_d_tensor, NA)
print("Output shape:", output.shape)


