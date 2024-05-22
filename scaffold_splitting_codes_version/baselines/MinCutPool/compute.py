import torch
import torch.nn.functional as F
from dataset import load_k_fold
from min_cut_model import MinCutPoolModel

from dataset_utils.chem_splitters import random_scaffold_split
from dataset_utils.utils import auc_metric
import pandas as pd
import os
from torch_geometric.data import DataLoader



def Train_Test_Baseline(graph_data, args):
    smiles_list = pd.read_csv(os.path.join(args.data_root, args.data_name, 'processed/smiles.csv'), header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = random_scaffold_split(graph_data, smiles_list, null_value=0,
                                                                       frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    scaffold_data = [train_loader, valid_loader, test_loader]

    if scaffold_data[-3].dataset.data.x is not None:
        avg_nodes = int(scaffold_data[-3].dataset.data.x.size(0) / len(scaffold_data[-3].dataset))
    else:
        avg_nodes = 50

    model = MinCutPoolModel(args.num_features, args.hidden_dim, args.num_classes, avg_nodes, args).to(args.device)

    optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs_test),
                                                           eta_min=args.learning_rate_min)
    criterions = [torch.nn.CrossEntropyLoss() for i in range(args.num_tasks)]

    print('###Scaffold, train/val/test:{},{},{}'.format(len(scaffold_data[-3].dataset), len(scaffold_data[-2].dataset),
                                                       len(scaffold_data[-1].dataset)))

    best_valid = 0.0
    best_valid_test = 0.0
    for epoch in range(1, args.epochs_test + 1):
        for data in scaffold_data[-3]:  # train dataloader
            model.train()

            data = data.to(args.device)
            pred, pool_loss = model(data)
            y = data.y.view(-1, args.num_tasks)

            loss = 0.0
            for i in range(args.num_tasks):
                is_valid = y[:, i] ** 2 > 0
                loss += criterions[i](pred[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())
            loss += pool_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        with torch.no_grad():
            y_true = []
            y_scores = []
            for data in scaffold_data[-2]:  # val dataloader
                model.eval()

                data = data.to(args.device)
                pred, _ = model(data)
                y = data.y.view(-1, args.num_tasks)

                y_true.append(y)
                y_scores.append(pred)
            valid_auc = auc_metric(y_true, y_scores)

        with torch.no_grad():
            y_true = []
            y_scores = []
            for data in scaffold_data[-1]:  # test dataloader
                model.eval()

                data = data.to(args.device)
                pred, _ = model(data)
                y = data.y.view(-1, args.num_tasks)

                y_true.append(y)
                y_scores.append(pred)
            test_auc = auc_metric(y_true, y_scores)
        if valid_auc >= best_valid:
            best_valid = valid_auc
            best_valid_test = test_auc

        if epoch % 10 == 0:
            print('epoch:{}, valid_auc:{:.4f}, test_auc:{:.4f}'.format(epoch, valid_auc, test_auc))

    return best_valid, best_valid_test

