import os
import numpy as np
import torch
import random
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from dataset_utils.chem_datasets import MoleculeDataset
from dataset_utils.chem_pre import pre_filter, pre_transform
from core.auto_model import AutoModel

graph_classification_dataset = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'hiv', 'muv']

def arg_parse():
    str2bool = lambda x: x.lower() == "true"

    parser = argparse.ArgumentParser("DAGC.")
    parser.add_argument('--data_name', type=str, default='clintox', help='the name of molecular graph dataset')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--fine_tune', type=str2bool, default=False, help='whether fine tune the optimal architecture')
    parser.add_argument('--hyper_epoch', type=int, default=20, help='it is valid when fine tune the optimal architecture')


    # child model, same as PAS
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--epochs_test', type=int, default=100, help='num of test epochs from scratch')
    parser.add_argument('--hidden_dim', type=int, default=64, help='default hidden_dim for gnn model')
    parser.add_argument('--dropout', type=float, default=0.0, help='default dropout for gnn model')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--optimizer', type=str, default='adam', help='default optimizer for child model')
    parser.add_argument('--postprocessing', type=str, default='att_mlp', help='the type of postprocessing(att_mlp/mlp)')

    # controller, the setting same as GraphNAS, No modification required
    parser.add_argument('--controller_train_epoch', type=int, default=200, help='train epoch of controller') ## same as PAS
    parser.add_argument('--search_scale', type=int, default=100, help='sample number of controller after trained')
    parser.add_argument('--test_gnn_num', type=int, default=5, help='test num of best GNN')
    parser.add_argument('--controller_lr', type=float, default=3.5e-4, help='learning rate of controller')
    parser.add_argument('--cuda', type=str2bool, default=True, help='train controller use cuda')
    parser.add_argument('--entropy_coeff', type=float, default=1e-4, help='entropy coeff of controller')
    parser.add_argument('--ema_baseline_decay', type=float, default=0.95, help='ema baseline decay  of controller')
    parser.add_argument('--discount', type=float, default=1.0, help='discount of controller')
    parser.add_argument('--controller_train_parallel_num', type=int, default=1, help='sample num of GNN in each training of controller')
    parser.add_argument('--controller_grad_clip', type=float, default=0.0, help='controller grad clip of controller')
    parser.add_argument('--tanh_c', type=float, default=2.5, help='tanh_c of controller')
    parser.add_argument('--softmax_temperature', type=float, default=5.0, help='softmax_temperature of controller')
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

device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
args.device = device

args.graph_classification_dataset = graph_classification_dataset
args.data_root = './chem_data'
if args.data_name in args.graph_classification_dataset:
    graph_data = MoleculeDataset(os.path.join(args.data_root, args.data_name), dataset=args.data_name, pre_filter=pre_filter, pre_transform=pre_transform)

    args.num_features = graph_data.num_features
    args.num_classes = graph_data.num_classes
    args.num_tasks = int(graph_data.data.num_tasks[0])

    AutoModel(graph_data, args)
