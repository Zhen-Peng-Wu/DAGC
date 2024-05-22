import os
# import os.path as osp
import sys
import time
# import glob
# import pickle
# import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
# import genotypes
import torch.utils
# import torch_geometric.transforms as T
# import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
# from torch import cat
from torch_geometric.data import DataLoader
# from torch.autograd import Variable
from model import NetworkGNN as Network
# from utils import gen_uniform_60_20_20_split, save_load_split
from dataset import load_data, load_k_fold
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, Reddit,PPI
# from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import add_self_loops
from logging_util import init_logger
import torch.nn.functional as F

from dataset_utils.chem_datasets import MoleculeDataset
from dataset_utils.chem_pre import pre_filter, pre_transform
from dataset_utils.chem_splitters import random_scaffold_split
from dataset_utils.utils import auc_metric
import pandas as pd
from torch_geometric.data import DataLoader
import os

def main(exp_args):
    global train_args
    train_args = exp_args

    tune_str = time.strftime('%Y%m%d-%H%M%S')
    train_args.save = 'logs/tune-{}-{}'.format(train_args.data, tune_str)
    if not os.path.exists(train_args.save):
        os.mkdir(train_args.save)

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)



    #np.random.seed(train_args.seed)
    torch.cuda.set_device(train_args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(train_args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    # np.random.seed(train_args.seed)
    # torch.backends.cudnn.deterministic = True

    # if train_args.data == 'Amazon_Computers':
    #     data = Amazon('../data/AmazonComputers', 'Computers')
    # elif train_args.data == 'Amazon_Photo':
    #     data = Amazon('../data/AmazonPhoto', 'Photo')
    # elif train_args.data == 'Coauthor_Physics':
    #     data = Coauthor('../data/CoauthorPhysics', 'Physics')
    #
    # elif train_args.data == 'Coauthor_CS':
    #     data = Coauthor('../data/CoauthorCS', 'CS')
    #
    # elif train_args.data == 'Cora_Full':
    #     dataset = CoraFull('../data/Cora_Full')
    # elif train_args.data == 'PubMed':
    #     data = Planetoid('../data/', 'PubMed')
    # elif train_args.data == 'Cora':
    #     data = Planetoid('../data/', 'Cora')
    # elif train_args.data == 'CiteSeer':
    #     data = Planetoid('../data/', 'CiteSeer')
    # elif train_args.data == 'PPI':
    #     train_dataset = PPI('../data/PPI', split='train')
    #     val_dataset = PPI('../data/PPI', split='val')
    #     test_dataset = PPI('../data/PPI', split='test')
    #     num_features = train_dataset.num_features
    #     num_classes = train_dataset.num_classes
    #     ppi_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #     ppi_val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    #     ppi_test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    #     print('load PPI done!')
    #     data = [ppi_train_loader, ppi_val_loader, ppi_test_loader]

    train_args.data_root = '../../chem_data'
    graph_data = MoleculeDataset(os.path.join(train_args.data_root, train_args.data), dataset=train_args.data,
                                 pre_filter=pre_filter, pre_transform=pre_transform)
    train_args.num_features = graph_data.num_features
    train_args.num_classes = graph_data.num_classes
    train_args.num_tasks = int(graph_data.data.num_tasks[0])

    smiles_list = pd.read_csv(os.path.join(train_args.data_root, train_args.data, 'processed/smiles.csv'), header=None)[
        0].tolist()
    train_dataset, valid_dataset, test_dataset = random_scaffold_split(graph_data, smiles_list, null_value=0,
                                                                       frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    train_loader = DataLoader(train_dataset, train_args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, train_args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, train_args.batch_size, shuffle=False)
    data = [graph_data, None, None, None, train_loader, valid_loader, test_loader]


    hidden_size = train_args.hidden_size


    genotype = train_args.arch
    # if train_args.data == 'PPI':
    #     criterion = nn.BCEWithLogitsLoss()
    #     criterion = criterion.cuda()
    # else:
    criterion = [torch.nn.CrossEntropyLoss() for i in range(train_args.num_tasks)]

    num_nodes = graph_data.data.x.size(0)
    model = Network(genotype, criterion, train_args.num_features, train_args.num_classes, hidden_size,
                    num_layers=train_args.num_layers, in_dropout=train_args.in_dropout, out_dropout=train_args.out_dropout,
                    act=train_args.activation, args = exp_args,is_mlp = train_args.is_mlp, num_nodes=num_nodes)
    model = model.cuda()

    logging.info("genotype=%s, param size = %fMB, args=%s", genotype, utils.count_parameters_in_MB(model), train_args.__dict__)
    print('param size = %fMB', utils.count_parameters_in_MB(model))
    def get_optimizer():
        if train_args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                train_args.learning_rate,
                # momentum=train_args.momentum,
                weight_decay=train_args.weight_decay
            )
        elif train_args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                train_args.learning_rate,
                momentum=train_args.momentum,
                weight_decay=train_args.weight_decay
            )
        elif train_args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(
                model.parameters(),
                train_args.learning_rate,
                weight_decay=train_args.weight_decay
            )
        return optimizer
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs))
    if train_args.ft_mode == 'kfold' and train_args.data in train_args.graph_classification_dataset:
        best_valid = 0.0
        best_valid_test = 0.0

        model.reset_params()
        optimizer = get_optimizer()
        if train_args.cos_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs),
                                                                   eta_min=train_args.lr_min)
        print('#####train/val/test:{},{},{}'.format(len(data[4].dataset), len(data[5].dataset),
                                                             len(data[5].dataset)))
        for epoch in range(train_args.epochs):
            train_auc, train_obj = train_graph(data, model, criterion, optimizer)
            if train_args.cos_lr:
                scheduler.step()

            valid_auc, valid_obj = infer_graph(data, model, criterion)
            test_auc, test_obj = infer_graph(data, model, criterion, test=True)

            if valid_auc >= best_valid:
                best_valid = valid_auc
                best_valid_test = test_auc

            if epoch % 10 == 0:
                logging.info('epoch=%s, lr=%s, train_obj=%s, train_auc=%f, valid_auc=%s', epoch,
                             scheduler.get_lr()[0] if train_args.cos_lr else train_args.learning_rate, train_obj,
                             train_auc, valid_auc)
                print(
                    'epoch={}, lr={}, train_obj={:.08f}, train_auc={:.04f}, valid_loss={:.08f},valid_auc={:.04f},test_auc={:.04f}'.format(
                         epoch, scheduler.get_lr()[0] if train_args.cos_lr else train_args.learning_rate,
                        train_obj, train_auc, valid_obj, valid_auc, test_auc))

            if train_args.show_info:
                print(
                    'epoch={}, lr={}, train_obj={:.08f}, train_auc={:.04f}, valid_loss={:.08f},valid_auc={:.04f},test_auc={:.04f}'.format(
                        epoch, scheduler.get_lr()[0] if train_args.cos_lr else train_args.learning_rate,
                        train_obj, train_auc, valid_obj, valid_auc, test_auc))

            utils.save(model, os.path.join(train_args.save, 'weights.pt'))
        return best_valid, best_valid_test, 0.0, train_args

    else: #split
        optimizer = get_optimizer()
        model.reset_params()
        min_valid_loss = float("inf")
        best_valid_acc = 0
        best_test_acc = 0

        if train_args.cos_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs), eta_min=0.0001)
        for epoch in range(train_args.epochs):
            train_acc, train_obj = train_graph(data, model, criterion, optimizer)

            if train_args.cos_lr:
                scheduler.step()

            valid_acc, valid_obj = infer_graph(data, model, criterion)
            test_acc, test_obj = infer_graph(data, model, criterion, test=True)
            if valid_obj < min_valid_loss:
                min_valid_loss = valid_obj
                best_valid_acc = valid_acc
                best_test_acc = test_acc
            if epoch % 10 == 0:
                logging.info('epoch=%s, lr=%s, train_obj=%s, train_acc=%f, valid_acc=%s',
                             epoch, scheduler.get_lr()[0] if train_args.cos_lr else train_args.learning_rate,
                             train_obj, train_acc, valid_acc)
            if train_args.show_info:
                print('epoch={}, lr={}, train_obj={:.08f}, train_acc={:.04f}, valid_loss={:.08f},valid_acc={:.04f},test_acc={:.04f}'.format(
                        epoch, scheduler.get_lr()[0] if train_args.cos_lr else train_args.learning_rate,
                        train_obj, train_acc, valid_obj, valid_acc, test_acc))

            utils.save(model, os.path.join(train_args.save, 'weights.pt'))

        return best_valid_acc, best_test_acc,  0, train_args


def train_graph(data, model,  criterion, model_optimizer):
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []

    # data:[dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader]
    for train_data in data[4]:

        train_data = train_data.to(device)
        model_optimizer.zero_grad()

        output = model(train_data).to(device)
        y = train_data.y.view(-1, train_args.num_tasks)
        y_true.append(y)
        y_pred.append(output)

        #error loss and resource loss
        if train_args.data =='COLORS-3':
            error_loss = criterion(output, train_data.y.long())
        else:
            error_loss = 0.0
            for i in range(train_args.num_tasks):
                is_valid = y[:, i] ** 2 > 0
                error_loss += criterion[i](output[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())

        total_loss += error_loss.item()

        error_loss.backward(retain_graph=True)
        model_optimizer.step()
    auc = auc_metric(y_true, y_pred)
    return auc, total_loss / len(data[4].dataset)

def infer_graph(data_, model, criterion, test=False):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    #for valid or test.
    if test:
        data = data_[6]
    else:
        data = data_[5]
    for val_data in data:
        val_data = val_data.to(device)
        with torch.no_grad():
            logits = model(val_data).to(device)
            y = val_data.y.view(-1, train_args.num_tasks)
            y_true.append(y)
            y_pred.append(logits)
        # target = val_data.y
        if train_args.data =='COLORS-3':
            loss = criterion(logits, target.long())
        else:
            loss = 0.0
            for i in range(train_args.num_tasks):
                is_valid = y[:, i] ** 2 > 0
                loss += criterion[i](logits[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())
        total_loss += loss.item()
        valid_auc = auc_metric(y_true, y_pred)
    return valid_auc, total_loss/len(data.dataset)

if __name__ == '__main__':
  main()


