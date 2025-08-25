import torch.nn as nn
import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GATemb(nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels, num_heads=3):
        super(GATemb, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.FC1 = torch.nn.Linear(hidden_channels* num_heads, 64)
        self.FC2 = torch.nn.Linear(64, out_channels)


    def forward(self, data):
        x,edge_index = data.x,data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x,training= self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.FC1(x)
        x = self.FC2(x)
        return x

