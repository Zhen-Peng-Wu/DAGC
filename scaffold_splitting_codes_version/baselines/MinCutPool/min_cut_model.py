from math import ceil
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DenseGraphConv, GCNConv, dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU


class MinCutPoolModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, avg_nodes, args):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        num_nodes = ceil(0.5 * avg_nodes)
        self.mlp1 = Linear(hidden_channels, num_nodes)
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        num_nodes = ceil(0.5 * num_nodes)
        self.mlp2 = Linear(hidden_channels, num_nodes)
        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)

        self.args = args
        self.num_tasks = self.args.num_tasks
        self.post_tasks = nn.ModuleList()
        for i in range(self.num_tasks):
            mlp = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, out_channels))
            self.post_tasks.append(mlp)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index).relu()
        
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)
        
        s = self.mlp1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)
        
        x = self.conv2(x, adj).relu()
        s = self.mlp2(x)
        
        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        
        x = self.conv3(x, adj)
        
        x = x.mean(dim=1)

        pred_tasks = []
        for i in range(self.num_tasks):
            logits = self.post_tasks[i](x)
            pred = F.log_softmax(logits, dim=-1)
            pred_tasks.append(pred)
        pred_tasks = torch.stack(pred_tasks, dim=-1)
        return pred_tasks, mc1 + mc2 + o1 + o2
