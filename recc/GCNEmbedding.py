import torch.nn
from torch_geometric.nn.conv import GCNConv
import os,sys
# 获取绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
parent_dir = os.path.dirname(script_dir)  # 上一级目录
# 将上一级目录添加到sys.path
sys.path.append(parent_dir)

from config import *
import torch.nn.functional as F


class GCNemb(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_dim,cluster_num):
        super(GCNemb, self).__init__()
        self.centers = torch.nn.Parameter(torch.randn(cluster_num , output_dim))
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        # self.FC1 = torch.nn.Linear(hidden_dim, 64)
        # self.FC2 = torch.nn.Linear(64, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training, p=args.dropout)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training, p=args.dropout)
        x = self.conv3(x, edge_index)
        # x = F.elu(x)
        # x = F.dropout(x, training=self.training, p=args.dropout)
        # x = self.FC1(x)
        # x = F.elu(x)
        # x = self.FC2(x)
        return x
#
#
