import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import conv_map, global_pooling_map
from torch.nn import Linear, Sequential, ReLU


class BaselineModel(torch.nn.Module):

    def __init__(self,
                 args):

        super(BaselineModel, self).__init__()

        self.args = args
        self.in_features = self.args.num_features
        self.num_classes = self.args.num_classes
        self.hidden_dim = self.args.hidden_dim
        self.layer_num = self.args.layer_num

        self.lin1 = nn.Linear(self.in_features, self.hidden_dim)

        self.convs = []
        for i in range(self.layer_num):
            conv = conv_map(self.args.baseline_name, self.hidden_dim, self.hidden_dim)
            self.add_module(f"conv{i}", conv)
            self.convs.append(conv)

        self.global_pooling = global_pooling_map(self.args.pooling_type)

        self.mlp = Sequential(
            Linear(self.hidden_dim, self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, data):

        edge_weight = torch.ones(data.edge_index.size()[1], device=data.edge_index.device).float()
        data.edge_weight = edge_weight

        data.x = F.relu(self.lin1(data.x))
        data.x = F.dropout(data.x, p=self.args.dropout, training=self.training)

        x = data.x
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index, edge_weight=data.edge_weight)

            x = F.relu(x)

            layer_norm = nn.LayerNorm(normalized_shape=x.size(), elementwise_affine=False)
            x = layer_norm(x)

            x = F.dropout(x, p=self.args.dropout, training=self.training)

        global_graph_emb = self.global_pooling(x, data.batch)

        scores = self.mlp(global_graph_emb)
        logits = F.log_softmax(scores, dim=-1)
        return logits

