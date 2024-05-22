import os
import numpy as np
import torch
import random
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from dataset_utils.chem_datasets import MoleculeDataset
from dataset_utils.chem_pre import pre_filter, pre_transform
from compute import Train_Test_Baseline


graph_classification_dataset = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'hiv', 'muv']


## the example of run command: python main.py --baseline_name GCNConv --data_name clintox
def arg_parse():
    parser = argparse.ArgumentParser("baseline.")
    parser.add_argument('--baseline_name', type=str, default='GCNConv',
                        help='convolution function for baseline, this need to specified')
    parser.add_argument('--data_name', type=str, default='clintox', help='location of the data corpus')

    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--layer_num', type=int, default=2, help='default the number of gnn layer')
    parser.add_argument('--pooling_type', type=str, default='global_add', help='default global pooling function.')
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
args.data_root = '../../chem_data'
if args.data_name in args.graph_classification_dataset:
    graph_data = MoleculeDataset(os.path.join(args.data_root, args.data_name), dataset=args.data_name,
                                 pre_filter=pre_filter, pre_transform=pre_transform)

    args.num_features = graph_data.num_features
    args.num_classes = graph_data.num_classes
    args.num_tasks = int(graph_data.data.num_tasks[0])


    ## train from scratch
    test_repeat = 20
    tests = []
    for i in range(test_repeat):
        valid_auc, test_auc = Train_Test_Baseline(graph_data, args)
        print("{}-th run in all {} runs || Test_auc:{:.4f}".format(i + 1, test_repeat, test_auc))
        tests.append(test_auc)
    tests = np.array(tests)
    print("Final results --- Test_auc:{:.4f}±{:.4f}".format(tests.mean(), tests.std()))
