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

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

### the example of run command: python main.py --dataset DD --gpu 0
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
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=100,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=5,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--folds', default=20, 		type=int, 			help='Cross validation folds')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda:%d' % args.gpu)

## get data
dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset)
dataset.data.edge_attr = None

# load and process
if dataset.data.x is None:
    max_degree = 0
    degs = []
    for data in dataset:
        degs += [degree(data.edge_index[0], dtype=torch.long)]
        max_degree = max(max_degree, degs[-1].max().item())

    if max_degree < 1000:
        dataset.transform = T.OneHotDegree(max_degree)
    else:
        deg = torch.cat(degs, dim=0).to(torch.float)
        mean, std = deg.mean().item(), deg.std().item()
        dataset.transform = NormalizedDegree(mean, std)

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

from sklearn.model_selection import KFold
# use k fold cross-validation
def k_fold(data, args):
    kf = KFold(args.folds, shuffle=True, random_state=args.seed)

    test_indices, train_indices = [], []
    for _, idx in kf.split(torch.zeros(len(data)), data.data.y):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(args.folds)]

    for i in range(args.folds):
        train_mask = torch.ones(len(data), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices

# num_training = int(len(dataset)*0.8)
# num_val = int(len(dataset)*0.1)
# num_test = len(dataset) - (num_training+num_val)
# training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])
#
# train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
# val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
# test_loader = DataLoader(test_set,batch_size=1,shuffle=False)


def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)

test_repeat = 5
for tr in range(test_repeat):
    # iterate over k folds
    test_acc_folds = []
    valid_acc_folds = []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args))):
        print('fold-{}'.format(fold+1))

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

        model = Net(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        min_loss = 1e10
        patience = 0

        for epoch in range(args.epochs):
            model.train()
            for i, data in enumerate(train_loader):
                data = data.to(args.device)
                out = model(data)
                loss = F.nll_loss(out, data.y)
                print("Training loss:{}".format(loss.item()))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            val_acc,val_loss = test(model,val_loader)
            print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
            valid_acc_folds.append(val_acc)
            # if val_loss < min_loss:
            #     torch.save(model.state_dict(),args.dataset+'_latest.pth')
            #     print("Model saved at epoch{}".format(epoch))
            #     min_loss = val_loss
            #     patience = 0
            # else:
            #     patience += 1
            # if patience > args.patience:
            #     break

            # model = Net(args).to(args.device)
            # model.load_state_dict(torch.load(args.dataset+'_latest.pth'))
            test_acc,test_loss = test(model,test_loader)
            print("Test accuarcy:{}".format(test_acc))
            test_acc_folds.append(test_acc)

    valid_accs = torch.tensor(valid_acc_folds).view(args.folds, args.epochs)
    test_accs = torch.tensor(test_acc_folds).view(args.folds, args.epochs)

    # max_valid_acc
    valid_accs, argmax = valid_accs.max(dim=-1)
    valid_acc_mean = round(valid_accs.mean().item() * 100, 2)
    test_accs_argmax = test_accs[torch.arange(args.folds, dtype=torch.long), argmax] * 100
    test_acc_mean = round(test_accs_argmax.mean().item(), 2)
    test_acc_std = round(test_accs_argmax.std().item(), 2)

    print("{}-th run in all {} runs || Test_acc:{}±{}".format(tr + 1, test_repeat, test_acc_mean, test_acc_std))
