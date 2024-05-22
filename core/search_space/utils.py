from torch_geometric.nn import GCNConv,GraphConv
from core.search_space.convolution import GATConvO,SAGEConvO,GINConvO,GeneralConvO
from torch.nn import Linear, Sequential, ReLU
from core.search_space.local_pooling import TopKPool,SAGPool,ASAPool,PANPool,HopPool,GAPPool
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
        conv_layer = SAGEConvO(input_dim, hidden_dim, normalize=True, aggr='add')
    elif conv_type == 'GINConv':
        nn1 = Sequential(Linear(input_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        conv_layer = GINConvO(nn=nn1, aggr='add')
    elif conv_type == 'GraphConv':
        conv_layer = GraphConv(input_dim, hidden_dim, aggr='add')
    elif conv_type == 'GeneralConv':
        conv_layer = GeneralConvO(input_dim, hidden_dim,skip_linear=True, aggr='add')
    else:
        raise Exception("Wrong conv function")
    return conv_layer

def local_pooling_map(local_pooling_type, hidden_dim):
    if local_pooling_type == 'TopKPool':
        local_pooling_layer = TopKPool(hidden_dim, min_score=-1)
    elif local_pooling_type == 'SAGPool':
        local_pooling_layer = SAGPool(hidden_dim, min_score=-1, GNN=GCNConv)
    elif local_pooling_type == 'ASAPool':
        local_pooling_layer = ASAPool(hidden_dim)
    elif local_pooling_type == 'PANPool':
        local_pooling_layer = PANPool(hidden_dim, min_score=-1)
    elif local_pooling_type == 'HopPool':
        local_pooling_layer = HopPool(hidden_dim)
    elif local_pooling_type == 'GCPool':
        local_pooling_layer = SAGPool(hidden_dim, min_score=-1, GNN=GraphConv)
    elif local_pooling_type == 'GAPPool':
        local_pooling_layer = GAPPool(hidden_dim)
    elif local_pooling_type == 'None':
        local_pooling_layer = None
    else:
        raise Exception("Wrong local_pooling function")
    return local_pooling_layer
