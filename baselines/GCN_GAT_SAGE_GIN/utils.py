from torch_geometric.nn import GCNConv
from convolution import GATConvO,SAGEConvO,GINConvO
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


def global_pooling_map(global_pooling_type):
    if global_pooling_type == 'global_max':
        global_pooling = global_max_pool
    elif global_pooling_type == 'global_mean':
        global_pooling = global_mean_pool
    elif global_pooling_type == 'global_add':
        global_pooling = global_add_pool
    else:
        raise Exception("Wrong global_pooling function")
    return global_pooling


def conv_map(conv_type, input_dim, hidden_dim):
    if conv_type == 'GCNConv':
        conv_layer = GCNConv(input_dim, hidden_dim, aggr='add')
    elif conv_type == 'GATConv':
        heads = 2
        conv_layer = GATConvO(input_dim, hidden_dim//heads, heads=heads, aggr='add')
    elif conv_type == 'SAGEConv':
        conv_layer = SAGEConvO(input_dim, hidden_dim, normalize=True, aggr='mean')
    elif conv_type == 'GINConv':
        nn1 = Sequential(Linear(input_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        conv_layer = GINConvO(nn=nn1, aggr='add')
    else:
        raise Exception("Wrong conv function")
    return conv_layer

