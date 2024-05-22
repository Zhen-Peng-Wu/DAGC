import sys
import numpy as np
import torch
import utils
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score
from datasets import load_data, load_k_fold
from model import NetworkGNN as Network
# from ogb.graphproppred import Evaluator

import logging
from sklearn.metrics import pairwise_distances
from torch_scatter import scatter_mean, scatter_sum

from dataset_utils.chem_datasets import MoleculeDataset
from dataset_utils.chem_pre import pre_filter, pre_transform
from dataset_utils.chem_splitters import random_scaffold_split
from dataset_utils.utils import auc_metric
import pandas as pd
from torch_geometric.data import DataLoader
import os
import torch.nn.functional as F


def main(exp_args, run=0):
    global train_args
    train_args = exp_args

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


    # data = data.to(device)
    train_args.data_root = '../../chem_data'
    graph_data = MoleculeDataset(os.path.join(train_args.data_root, train_args.data), dataset=train_args.data,
                                 pre_filter=pre_filter, pre_transform=pre_transform)
    train_args.num_features = graph_data.num_features
    train_args.num_classes = graph_data.num_classes
    train_args.num_tasks = int(graph_data.data.num_tasks[0])

    smiles_list = pd.read_csv(os.path.join(train_args.data_root, train_args.data, 'processed/smiles.csv'), header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = random_scaffold_split(graph_data, smiles_list, null_value=0,
                                                                       frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    train_loader = DataLoader(train_dataset, train_args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, train_args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, train_args.batch_size, shuffle=False)
    data = [graph_data, None, None, None, train_loader, valid_loader, test_loader]

    if 'ogb' in train_args.data:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = [torch.nn.CrossEntropyLoss() for i in range(train_args.num_tasks)]
    genotype = train_args.arch
    hidden_size = train_args.hidden_size


    model = Network(genotype, criterion, train_args.num_features, train_args.num_classes, hidden_size, dropout=train_args.dropout,args=train_args)

    model = model.cuda()
    num_parameters = np.sum(np.prod(v.size()) for name, v in model.named_parameters())
    print('params size:', num_parameters)
    logging.info("genotype=%s, param size = %fMB, args=%s", genotype, utils.count_parameters_in_MB(model), train_args.__dict__)

    if train_args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            train_args.learning_rate,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs))

    if train_args.ft_mode == 'kfold':

        best_valid = 0.0
        best_valid_test = 0.0

        model.reset_parameters()
        if train_args.cos_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs),
                                                                   eta_min=train_args.lr_min)

        for epoch in range(train_args.epochs):
            train_auc, train_loss = train(data, model, criterion, optimizer)
            if train_args.cos_lr:
                scheduler.step()

            valid_auc, valid_loss = infer(data, model, criterion)
            test_auc, test_loss = infer(data, model, criterion, test=True)

            if valid_auc >= best_valid:
                best_valid = valid_auc
                best_valid_test = test_auc

            if epoch % 10 == 0:
                logging.info(
                    'epoch=%s, lr=%s, train_loss=%s, train_auc=%f, valid_loss=%f, valid_auc=%s, test_loss=%s, test_auc=%s',
                    epoch,
                    scheduler.get_lr()[0] if train_args.cos_lr else train_args.learning_rate, train_loss, train_auc,
                    valid_loss, valid_auc, test_loss, test_auc)
                print(
                    'epoch={}, lr={}, train_obj={:.08f}, train_auc={:.04f}, valid_loss={:.08f},valid_auc={:.04f},test_auc={:.04f}'.format(
                        epoch, scheduler.get_lr()[0] if train_args.cos_lr else train_args.learning_rate,
                        train_loss, train_auc, valid_loss, valid_auc, test_auc))


                # utils.save(model, os.path.join(train_args.save, 'weights.pt'))
        return best_valid, best_valid_test, 0.0, train_args

    else: #split_one
        model.reset_parameters()
        min_valid_loss = float("inf")
        # max_valid_acc = 0
        best_valid_acc = 0
        best_test_acc = 0
        best_epoch = 0
        valid_losses = []
        valid_accs = []
        test_accs = []

        if train_args.cos_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs), eta_min=train_args.lr_min)

        for epoch in range(train_args.epochs):
            train_acc, train_loss = train(data, model, criterion, optimizer)
            if train_args.cos_lr:
                scheduler.step()
            valid_acc, valid_loss = infer(data, model, criterion)
            test_acc, test_loss = infer(data, model, criterion, test=True)
            valid_accs.append(valid_acc)
            valid_losses.append(valid_loss)
            test_accs.append(test_acc)
            if 'ogb' in train_args.data:
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    best_valid_acc = valid_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
                # if valid_acc > best_valid_acc:
                #     best_valid_acc = valid_acc
                #     best_test_acc = test_acc
                #     best_epoch = epoch
            else:
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_test_acc = test_acc
                    best_epoch = epoch

            if epoch % 10 == 0 or 'ogb' in train_args.data: 
                logging.info('epoch=%s, lr=%s, train_loss=%s, train_acc=%f, valid_loss=%s, valid_acc=%s, '
                             'test_loss=%s, test_acc=%s, best_valid_acc=%f, best_test_acc=%f', epoch,
                             scheduler.get_lr()[0] if train_args.cos_lr else train_args.learning_rate,
                             train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc, best_valid_acc, best_test_acc)

        print('final: epoch:{}, valid_loss={:.08f},valid_acc={:.04f},test_acc={:.04f}'.format(
            best_epoch, valid_losses[best_epoch], valid_accs[best_epoch], test_accs[best_epoch]))
        return best_valid_acc, best_test_acc,  0, train_args

def train(data, model,  criterion, model_optimizer):
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []
    # data:[dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader]
    for train_data in data[4]:

        train_data = train_data.to(device)
        model_optimizer.zero_grad()

        output = model(train_data)
        y = train_data.y.view(-1, train_args.num_tasks)
        y_true.append(y)
        y_pred.append(output)

        if 'ogb' in train_args.data:
            is_labeled = ((train_data.y == train_data.y)&(output==output))
            error_loss = criterion(output.to(torch.float32)[is_labeled], train_data.y.to(torch.float32)[is_labeled])
            y_true.append(train_data.y[is_labeled].view(output[is_labeled].shape).detach().cpu())
            y_pred.append(output[is_labeled].detach().cpu())
        else:
            error_loss = 0.0
            for i in range(train_args.num_tasks):
                is_valid = y[:, i] ** 2 > 0
                error_loss += criterion[i](output[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())

        total_loss += error_loss.item()

        error_loss.backward()
        model_optimizer.step()

    if 'ogb' in train_args.data:
        evaluator = Evaluator(train_args.data)
        y_true = torch.cat(y_true, dim=0).numpy().reshape(-1, 1)
        y_pred = torch.cat(y_pred, dim=0).numpy().reshape(-1, 1)
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        try:
            train_acc = evaluator.eval(input_dict)['rocauc']
            return train_acc, total_loss / len(data[4].dataset)
        except RuntimeError as e:
            return 0, float("inf")
    else:
        auc = auc_metric(y_true, y_pred)
        return auc, total_loss / len(data[4].dataset)

def infer(data_, model, criterion, test=False):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    results = []
    #for valid or test.
    if test:
        data = data_[6]
    else:
        data = data_[5]

    for val_data in data:
        val_data = val_data.to(device)
        with torch.no_grad():
            logits = model(val_data)
            y = val_data.y.view(-1, train_args.num_tasks)
            y_true.append(y)
            y_pred.append(logits)

        target = val_data.y

        if 'ogb' in train_args.data:
            loss = criterion(logits.to(torch.float32), target.to(torch.float32))
            y_true.append(target.view(logits.shape).detach().cpu())
            y_pred.append(logits.detach().cpu())
        else:
            loss = 0.0
            for i in range(train_args.num_tasks):
                is_valid = y[:, i] ** 2 > 0
                loss += criterion[i](logits[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())

        total_loss += loss.item()

    if 'ogb' in train_args.data:
        evaluator = Evaluator(train_args.data)
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        return evaluator.eval(input_dict)['rocauc'], total_loss / len(data.dataset)
    else:
        valid_auc = auc_metric(y_true, y_pred)
        return valid_auc, total_loss/len(data.dataset)


if __name__ == '__main__':
    main()


