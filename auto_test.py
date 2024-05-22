import os
import numpy as np
import torch
import random
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from dataset import load_data
from core.estimation import Scratch_Train_Test

graph_classification_dataset=['DD','MUTAG','PROTEINS','NCI1','NCI109','IMDB-BINARY','REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI', 'COLORS-3', 'COLLAB', 'REDDIT-MULTI-5K']

def arg_parse():
    str2bool = lambda x: x.lower() == "true"

    parser = argparse.ArgumentParser("DAGC.")
    parser.add_argument('--data_name', type=str, default='MUTAG', help='location of the data corpus')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

    # child model
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.0005, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--epochs_test', type=int, default=100, help='num of test epochs from scratch')
    parser.add_argument('--hidden_dim', type=int, default=64, help='default hidden_dim for gnn model')
    parser.add_argument('--dropout', type=float, default=0.0, help='default dropout for gnn model')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
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

def get_test_architecture(data_name):
    if data_name == "COX2":
        target_architecture = [9, 'SAGEConv', 'HopPool', 'global_add']
    elif data_name == "MUTAG":
        target_architecture = [4, 'GeneralConv', 'PANPool', 'global_add']
    elif data_name == "PROTEINS":
        target_architecture = [4, 'GINConv', 'GCPool', 'global_mean']
    elif data_name == "DD":
        target_architecture = [2, 'GCNConv', 'PANPool', 'global_add']
    elif data_name == "NCI109":
        target_architecture = [8, 'SAGEConv', 'None', 'global_max']
    else:
        raise Exception("Wrong dataset name")
    return target_architecture

args.graph_classification_dataset = graph_classification_dataset
if args.data_name in args.graph_classification_dataset:
    graph_data, num_nodes = load_data(args.data_name, batch_size=args.batch_size, split_seed=args.seed)
    num_features = graph_data[0].num_features
    if args.data_name == 'COLORS-3':
        num_classes = 11
    else:
        num_classes = graph_data[0].num_classes
    args.num_features = num_features
    args.num_classes = num_classes

    target_architecture = get_test_architecture(args.data_name)
    print(35 * "=" + " the testing start " + 35 * "=")
    print("dataset:", args.data_name)
    print("test gnn architecture", target_architecture)

    ## train from scratch
    test_repeat = 5
    for i in range(test_repeat):
        valid_acc, test_acc, test_acc_std = Scratch_Train_Test(target_architecture, graph_data, args=args)
        print("{}-th run in all {} runs || Test_acc:{}±{}".format(i + 1, test_repeat, test_acc, test_acc_std))
    print(35 * "=" + " the testing ending " + 35 * "=")

