import os
import configparser
from autognas.auto_model import AutoModel
from dataset import load_data
import argparse
import numpy as np
import torch
import random
from dataset_utils.chem_datasets import MoleculeDataset
from dataset_utils.chem_pre import pre_filter, pre_transform

## the example of run command: python graph_classification.py --data_name clintox
def arg_parse():
    parser = argparse.ArgumentParser("baseline.")
    parser.add_argument('--data_name', type=str, default='clintox', help='location of the data corpus')
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

    args.data_root = '../../chem_data'
    graph_data = MoleculeDataset(os.path.join(args.data_root, args.data_name), dataset=args.data_name,
                                 pre_filter=pre_filter, pre_transform=pre_transform)

    graph_data.data_root = args.data_root
    graph_data.data_name = args.data_name
    graph_data.num_tasks = int(graph_data.data.num_tasks[0])
    graph_data.batch_size = search_parameter['batch_size']

    gnn_parameter['data_name'] = args.data_name
    AutoModel(graph_data, search_parameter, gnn_parameter, args)
