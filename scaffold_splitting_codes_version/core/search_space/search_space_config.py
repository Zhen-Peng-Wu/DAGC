class SearchSpace(object):
    """
    Loading the search space dict, config the number of stack gcn layer

    Returns:
        component_value_dict:dict
            the search space component dict, the key is the stack gcn
            component and the value is the corresponding value
    """

    def __init__(self):
        self.stack_gcn_architecture = ['gnn_layers','conv','local_pooling','global_pooling']

        self.space_dict = {
            'gnn_layers': [1,2,3, 4, 5,6,7, 8, 9,10],
            'conv': ['GCNConv', 'GATConv','SAGEConv','GINConv','GraphConv','GeneralConv'],
            'local_pooling': ['TopKPool','SAGPool','ASAPool','PANPool','HopPool', 'GCPool', 'GAPPool', 'None'],
            'global_pooling': ['global_max', 'global_mean', 'global_add']
        }
