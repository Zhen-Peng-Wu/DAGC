import torch
import torch.nn.functional as F
import torch.nn as nn
from core.search_space.utils import conv_map, local_pooling_map, global_pooling_map
from torch.nn import Linear, Sequential, ReLU

class GnnModel(torch.nn.Module):
    """
    Constructing the stack gcn model based on stack gcn architecture,
    realizing the stack gcn model forward process.

    Args:
        architecture: list
            the stack gcn architecture describe
        in_features: int
            the original input dimension for the stack gcn model

    Returns:
        output: tensor
            the output of the stack gcn model.
    """
    def __init__(self,
                 architecture,
                 args):

        super(GnnModel, self).__init__()

        self.cells_num = architecture[0]
        self.architecture = architecture[1:]
        self.args = args

        self.in_features = self.args.num_features
        self.num_classes = self.args.num_classes
        self.hidden_dim = self.args.hidden_dim

        self.lin1 = nn.Linear(self.in_features, self.hidden_dim)
        self.init_global_pooling = global_pooling_map(self.architecture[2])

        self.cells = []
        for i in range(self.cells_num):
            cell = LayerNAS_Cell(self.architecture, self.hidden_dim, self.hidden_dim, self.args)
            self.add_module(f"cell{i}", cell)
            self.cells.append(cell)

        self.post_processing = Post_Process(self.hidden_dim, self.cells_num, self.num_classes, self.args)

        
    def forward(self, data):
        reprs = []

        edge_weight = torch.ones(data.edge_index.size()[1], device=data.edge_index.device).float()
        data.edge_weight = edge_weight

        data.x = F.relu(self.lin1(data.x))
        reprs.append(self.init_global_pooling(data.x, data.batch))


        data.x = F.dropout(data.x, p=self.args.dropout, training=self.training)


        for i, cell in enumerate(self.cells):
            data, r = cell(data)
            reprs.append(r)

        reprs = torch.stack(reprs, dim=-1) # (N, hidden_dim, L+1), N is the sample number in batch, L is the searched layer
        logits = self.post_processing(reprs)
        return logits


class LayerNAS_Cell(nn.Module):
    def __init__(self, architecture, in_features, hidden_dim, args):
        super().__init__()
        self.architecture = architecture
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.args = args

        self.conv = conv_map(architecture[0], in_features, hidden_dim)
        self.local_pooling = local_pooling_map(architecture[1], hidden_dim)
        self.global_pooling = global_pooling_map(architecture[2])

    def forward(self, data):
        ## conv
        x = self.conv(data.x, data.edge_index, edge_weight=data.edge_weight)

        x = F.relu(x)

        x = F.dropout(x, p=self.args.dropout, training=self.training)

        # local_pooling
        if self.local_pooling:
            x, data.edge_index, data.edge_weight, data.batch = self.local_pooling(x, data.edge_index, batch=data.batch, edge_weight=data.edge_weight)[:4]

        # global_pooling
        global_graph_emb = self.global_pooling(x, data.batch)

        data.x = x

        return data, global_graph_emb


class Post_Process(nn.Module):
    def __init__(self, hidden_dim, cells_num, num_classes, args):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cells_num = cells_num
        self.num_classes = num_classes
        self.args = args

        self.attention_layer = nn.Parameter(torch.zeros(self.cells_num + 1))
        nn.init.xavier_uniform_(self.attention_layer.view(*self.attention_layer.shape, -1))
        self.classfier = Sequential(
            Linear(self.hidden_dim, self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, self.num_classes)
        )

        self.mlp = Sequential(
            Linear(self.hidden_dim * (self.cells_num + 1), self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, reprs):
        if self.args.postprocessing == 'att_mlp':
            reprs = reprs @ self.attention_layer
            scores = self.classfier(reprs)
        else:
            reprs = reprs.view(-1, self.hidden_dim * (self.cells_num + 1))
            scores = self.mlp(reprs)

        logits = F.log_softmax(scores, dim=-1)
        return logits
