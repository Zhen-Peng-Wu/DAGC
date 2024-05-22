import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from networks import  Net
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import random_split

from torch_geometric.utils import degree
import torch_geometric.transforms as T

from dataset_utils.chem_datasets import MoleculeDataset
from dataset_utils.chem_pre import pre_filter, pre_transform
import numpy as np
from dataset_utils.chem_splitters import random_scaffold_split
from dataset_utils.utils import auc_metric
import pandas as pd
import os
from torch_geometric.data import DataLoader


### the example of run command: python main.py --dataset clintox --gpu 0
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='clintox',
                    help='bace/bbbp/clintox/sider/tox21/toxcast/hiv/muv')
parser.add_argument('--epochs', type=int, default=100,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=5,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='bace/bbbp/clintox/sider/tox21/toxcast/hiv/muv')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--folds', default=20, 		type=int, 			help='Cross validation folds')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda:%d' % args.gpu)


args.data_root = '../../chem_data'
args.data_name = args.dataset
graph_data = MoleculeDataset(os.path.join(args.data_root, args.data_name), dataset=args.data_name,
                                 pre_filter=pre_filter, pre_transform=pre_transform)

args.num_features = graph_data.num_features
args.num_classes = graph_data.num_classes
args.num_tasks = int(graph_data.data.num_tasks[0])




def test(model, loader):
    model.eval()
    y_true = []
    y_scores = []
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        y = data.y.view(-1, args.num_tasks)

        y_true.append(y)
        y_scores.append(out)
    auc = auc_metric(y_true, y_scores)
    return auc

## train from scratch
test_repeat = 20
tests = []

for tr in range(test_repeat):
    smiles_list = pd.read_csv(os.path.join(args.data_root, args.data_name, 'processed/smiles.csv'), header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = random_scaffold_split(graph_data, smiles_list, null_value=0,
                                                                       frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    best_valid = 0.0
    best_valid_test = 0.0

    model = Net(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterions = [torch.nn.CrossEntropyLoss() for i in range(args.num_tasks)]

    for epoch in range(args.epochs):
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            out = model(data)
            y = data.y.view(-1, args.num_tasks)

            loss = 0.0
            for i in range(args.num_tasks):
                is_valid = y[:, i] ** 2 > 0
                loss += criterions[i](out[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        valid_auc = test(model, valid_loader)

        test_auc = test(model, test_loader)

        if valid_auc >= best_valid:
            best_valid = valid_auc
            best_valid_test = test_auc
        if epoch % 10 == 0:
            print('epoch:{}, valid_auc:{:.4f}, test_auc:{:.4f}'.format(epoch, valid_auc, test_auc))

    test_auc = best_valid_test

    print("{}-th run in all {} runs || Test_auc:{:.4f}".format(tr + 1, test_repeat, test_auc))
    tests.append(test_auc)
tests = np.array(tests)
print("Final results --- Test_auc:{:.4f}±{:.4f}".format(tests.mean(), tests.std()))
