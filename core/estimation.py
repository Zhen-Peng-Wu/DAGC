import torch
from core.model.gnn_model import GnnModel
from core.model.logger import gnn_architecture_performance_save
import torch.nn.functional as F
from dataset import load_k_fold

def train_infer_model(data, args, model):
    data = data.to(args.device)
    pred = model(data)

    if args.data_name == 'COLORS-3':
        label = data.y.long()
    else:
        label = data.y.view(-1)
    return pred, label


def Estimation(gnn_architecture,
               graph_data,
               args,
               device = "cuda:0"
               ):

    model = GnnModel(gnn_architecture, args).to(device)

    optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    loss_fn = F.nll_loss

    return_performance = 0
    for epoch in range(1, int(args.epochs) + 1):
        for data in graph_data[-3]: # train dataloader
            model.train()

            probas_pred, ground_truth = train_infer_model(data, args, model)
            loss = loss_fn(probas_pred, ground_truth)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        scheduler.step()

        valid_acc = 0
        with torch.no_grad():
            for data in graph_data[-2]: # val dataloader
                model.eval()
                probas_pred, ground_truth = train_infer_model(data, args, model)
                valid_acc += probas_pred.max(1)[1].eq(ground_truth.view(-1)).sum().item()

            valid_acc /= len(graph_data[-2].dataset)
            return_performance = valid_acc
    gnn_architecture_performance_save(gnn_architecture, return_performance, args.data_name)
    return return_performance


def Scratch_Train_Test(gnn_architecture, graph_data, args):
    valid_accs = []
    test_accs = []


    folds = 20
    k_folds_data = load_k_fold(graph_data[0], folds, args.batch_size)
    argmax_list = []
    for fold, fold_data in enumerate(k_folds_data):

        model = GnnModel(gnn_architecture, args).to(args.device)

        optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs_test),
                                                               eta_min=args.learning_rate_min)
        loss_fn = F.nll_loss

        print('###fold {}, train/val/test:{},{},{}'.format(fold+1, len(fold_data[-3].dataset),len(fold_data[-2].dataset),len(fold_data[-1].dataset)))
        max_acc = 0#
        max_index = 0#
        for epoch in range(1, args.epochs_test + 1):
            for data in fold_data[-3]:  # train dataloader
                model.train()

                probas_pred, ground_truth = train_infer_model(data, args, model)
                loss = loss_fn(probas_pred, ground_truth)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
            scheduler.step()

            valid_acc = 0
            with torch.no_grad():
                for data in fold_data[-2]:  # val dataloader
                    model.eval()
                    probas_pred, ground_truth = train_infer_model(data, args, model)
                    valid_acc += probas_pred.max(1)[1].eq(ground_truth.view(-1)).sum().item()

                valid_acc /= len(fold_data[-2].dataset)
                valid_accs.append(valid_acc)

            test_acc = 0
            with torch.no_grad():
                for data in fold_data[-1]:  # test dataloader
                    model.eval()
                    probas_pred, ground_truth = train_infer_model(data, args, model)
                    test_acc += probas_pred.max(1)[1].eq(ground_truth.view(-1)).sum().item()

                test_acc /= len(fold_data[-1].dataset)
                test_accs.append(test_acc)
            if valid_acc >= max_acc:
                max_acc = valid_acc
                max_index = epoch-1

            if epoch % 10 == 0:
                print('fold:{}, epoch:{}, valid_acc:{:.4f}, test_acc:{:.4f}'.format(fold+1,epoch,valid_acc,test_acc))
        argmax_list.append(max_index)

    valid_accs = torch.tensor(valid_accs).view(folds, args.epochs_test)
    test_accs = torch.tensor(test_accs).view(folds, args.epochs_test)

    # max_valid_acc
    valid_accs_argmax = valid_accs[torch.arange(folds, dtype=torch.long), argmax_list] * 100
    valid_acc_mean = round(valid_accs_argmax.mean().item(), 2)
    test_accs_argmax = test_accs[torch.arange(folds, dtype=torch.long), argmax_list] * 100
    test_acc_mean = round(test_accs_argmax.mean().item(), 2)
    test_acc_std = round(test_accs_argmax.std().item(), 2)
    print('test_accs:', test_accs_argmax)


    return valid_acc_mean, test_acc_mean, test_acc_std
