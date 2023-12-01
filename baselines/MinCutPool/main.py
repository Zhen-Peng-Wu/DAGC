import os
import numpy as np
import torch
import random
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from dataset import load_data
from compute import Train_Test_Baseline


graph_classification_dataset=['DD','MUTAG','PROTEINS','NCI1','NCI109','IMDB-BINARY','REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI', 'COLORS-3', 'COLLAB', 'REDDIT-MULTI-5K']

## the example of run command: python main.py --data_name DD --gpu 0
def arg_parse():
    parser = argparse.ArgumentParser("MinCutPool.")
    parser.add_argument('--data_name', type=str, default='COX2', help='location of the data corpus')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.0005, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--epochs_test', type=int, default=100, help='num of test epochs from scratch')
    parser.add_argument('--hidden_dim', type=int, default=64, help='default hidden_dim for gnn model')
    parser.add_argument('--dropout', type=float, default=0.0, help='default dropout for gnn model')
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

device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
args.device = device

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

    ## train from scratch
    test_repeat = 5
    for i in range(test_repeat):
        valid_acc, test_acc, test_acc_std = Train_Test_Baseline(graph_data, args)

        print("{}-th run in all {} runs || Test_acc:{}±{}".format(i + 1, test_repeat, test_acc, test_acc_std))
