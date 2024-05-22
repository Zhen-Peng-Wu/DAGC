import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import pandas as pd
import os
from core.model.gnn_model import GnnModel
from core.model.logger import gnn_architecture_performance_save
from dataset_utils.chem_splitters import random_scaffold_split
from dataset_utils.utils import auc_metric
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK
from copy import deepcopy

def train_one_epoch(train_data, model, optimizer, criterions, args):
    for data in train_data:
        model.train()

        data = data.to(args.device)
        pred = model(data)
        y = data.y.view(-1, args.num_tasks)

        loss = 0.0
        for i in range(args.num_tasks):
            is_valid = y[:, i] ** 2 > 0
            loss += criterions[i](pred[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_func(eval_data, model, args):
    with torch.no_grad():
        y_true = []
        y_scores = []
        for data in eval_data:
            model.eval()

            data = data.to(args.device)
            pred = model(data)
            y = data.y.view(-1, args.num_tasks)

            y_true.append(y)
            y_scores.append(pred)
        eval_auc = auc_metric(y_true, y_scores)
        return eval_auc

def Estimation(gnn_architecture,
               scaffold_data,
               args,
               device = "cuda:0"
               ):

    model = GnnModel(gnn_architecture, args).to(device)

    optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate/10.0)

    criterions = [F.nll_loss for i in range(args.num_tasks)]

    return_performance = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_one_epoch(scaffold_data[-3], model, optimizer, criterions, args)
        scheduler.step()

        valid_auc = eval_func(scaffold_data[-2], model, args)
        return_performance = valid_auc
    gnn_architecture_performance_save(gnn_architecture, return_performance, args.data_name)
    return return_performance




def fine_tune_scaffold_gnn(gnn_architecture, graph_data, args):

    def objective(hp_dict):
        current_args = deepcopy(args)
        for k, v in hp_dict.items():
            setattr(current_args, k, v)

        print('current_hyper:', current_args)
        valid_auc, test_auc = Scratch_Train_Test(gnn_architecture, graph_data, current_args)
        return {
            'loss': -valid_auc,
            'status': STATUS_OK
        }

    hyper_space = {
                   'hidden_dim': hp.choice('hidden_dim', [32, 64, 128]),
                   'learning_rate': hp.uniform("learning_rate", 0.001, 0.01),
                   'weight_decay': hp.uniform("weight_decay", 0.0001, 0.001),
                   'optimizer': hp.choice('optimizer', ['adagrad', 'adam']),
                   'dropout': hp.choice('dropout', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
                   }
    trials = Trials()
    best = fmin(objective, hyper_space, algo=partial(tpe.suggest, n_startup_jobs=int(args.hyper_epoch / 5)),
                max_evals=args.hyper_epoch, trials=trials)

    best_hp_dict = hyperopt.space_eval(hyper_space, best)

    best_args = deepcopy(args)
    for k, v in best_hp_dict.items():
        setattr(best_args, k, v)

    return best_args


def Scratch_Train_Test(gnn_architecture, graph_data, args):

    smiles_list = pd.read_csv(os.path.join(args.data_root, args.data_name, 'processed/smiles.csv'), header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = random_scaffold_split(graph_data, smiles_list, null_value=0,
                                                                frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)


    scaffold_data = [train_loader, valid_loader, test_loader]

    model = GnnModel(gnn_architecture, args).to(args.device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad([{'params': model.parameters()}],
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs_test), eta_min=args.learning_rate/10.0)
    criterions = [F.nll_loss for i in range(args.num_tasks)]

    print('###Scaffold, train/val/test:{},{},{}'.format(len(scaffold_data[-3].dataset), len(scaffold_data[-2].dataset),
                                                       len(scaffold_data[-1].dataset)))

    best_valid = 0.0
    best_valid_test = 0.0

    for epoch in range(1, args.epochs_test + 1):
        train_one_epoch(scaffold_data[-3], model, optimizer, criterions, args)
        scheduler.step()

        valid_auc = eval_func(scaffold_data[-2], model, args)
        test_auc = eval_func(scaffold_data[-1], model, args)

        if valid_auc >= best_valid:
            best_valid = valid_auc
            best_valid_test = test_auc

        if epoch % 10 == 0:
            print('epoch:{}, valid_auc:{:.4f}, test_auc:{:.4f}'.format(epoch, valid_auc, test_auc))

    return best_valid, best_valid_test
