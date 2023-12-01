import os
import configparser
from autognas.auto_model import AutoModel
from dataset import load_data
import argparse
import numpy as np
import torch
import random

## the example of run command: python graph_classification.py --data_name COX2
def arg_parse():
    parser = argparse.ArgumentParser("baseline.")
    parser.add_argument('--data_name', type=str, default='COX2', help='location of the data corpus')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = arg_parse()
set_seed(args.seed)



config = configparser.ConfigParser()

config_path = os.path.abspath(os.getcwd()) + "/config/graph_classification_config/"

configuration = ["graphnas.ini"]

for sub_config in configuration:
    config.read(config_path+sub_config)
    search_parameter = dict(config.items('search_parameter'))
    gnn_parameter = dict(config.items("gnn_parameter"))
    graph, num_nodes = load_data(args.data_name, batch_size=int(search_parameter['batch_size']), split_seed=args.seed)
    gnn_parameter['data_name'] = args.data_name
    AutoModel(graph, search_parameter, gnn_parameter)
