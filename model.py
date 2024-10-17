import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------- This part of the code is just for practice for now --------------------------------------------------

# Define 1st GCN layer to capture the spatial relationships between joints ()
class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_channels * 25, out_channels)  

    def forward(self, x, A):
        out = torch.einsum('nctv,vw->nctw', (x, A))  # Apply adjacency matrix
        N, C, T, V = out.shape  # (Batch size, Channels, Time steps, Joints)
        out = out.view(N, C * V, T)  # Combine channels and joints for linear transformation
        # Apply the linear transformation
        out = self.fc(out.transpose(1, 2))  
        return out.transpose(1, 2)  # Restore the original dimensions



# Full GCN Model to capture spatial and temporal features
class GCNModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(GCNModel, self).__init__()
        self.gcn1 = GCNLayer(in_channels, out_channels)  # First GCN layer
        self.conv1 = nn.Conv2d(out_channels, 64, kernel_size=(1, 9), padding=(0, 4))  # Temporal Conv layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)  # Fully connected layer for classification

    def forward(self, x, A):
        x = self.gcn1(x, A)  # Apply first GCN layer
        x = F.relu(x)  # activation (ReLU)
        x = x.unsqueeze(2)  

        # Apply temporal convolution
        x = self.conv1(x)
        x = F.relu(x)

        # Pooling and final classification
        x = self.pool(x)
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x


