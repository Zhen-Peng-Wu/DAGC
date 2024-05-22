import torch
import torch.nn.functional as F
from dataset import load_k_fold
from min_cut_model import MinCutPoolModel


def train_infer_model(data, args, model):
    data = data.to(args.device)
    pred, pool_loss = model(data)

    if args.data_name == 'COLORS-3':
        label = data.y.long()
    else:
        label = data.y.view(-1)
    return pred, label, pool_loss


def Train_Test_Baseline(graph_data, args):
    valid_accs = []
    test_accs = []

    folds = 20
    k_folds_data = load_k_fold(graph_data[0], folds, args.batch_size)

    for fold, fold_data in enumerate(k_folds_data):

        if fold_data[-3].dataset.data.x is not None:
            avg_nodes = int(fold_data[-3].dataset.data.x.size(0) / len(fold_data[-3].dataset))
        else:
            avg_nodes = 50

        model = MinCutPoolModel(args.num_features, args.hidden_dim, args.num_classes, avg_nodes).to(args.device)

        optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs_test),
                                                               eta_min=args.learning_rate_min)
        loss_fn = F.nll_loss

        print('###fold {}, train/val/test:{},{},{}'.format(fold+1, len(fold_data[-3].dataset),len(fold_data[-2].dataset),len(fold_data[-1].dataset)))
        for epoch in range(1, args.epochs_test + 1):
            for data in fold_data[-3]:  # train dataloader
                model.train()

                probas_pred, ground_truth, pool_loss = train_infer_model(data, args, model)
                loss = loss_fn(probas_pred, ground_truth)
                loss += pool_loss

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
            scheduler.step()

            valid_acc = 0
            with torch.no_grad():
                for data in fold_data[-2]:  # val dataloader
                    model.eval()
                    probas_pred, ground_truth, pool_loss = train_infer_model(data, args, model)
                    valid_acc += probas_pred.max(1)[1].eq(ground_truth.view(-1)).sum().item()

                valid_acc /= len(fold_data[-2].dataset)
                valid_accs.append(valid_acc)

            test_acc = 0
            with torch.no_grad():
                for data in fold_data[-1]:  # test dataloader
                    model.eval()
                    probas_pred, ground_truth, pool_loss = train_infer_model(data, args, model)
                    test_acc += probas_pred.max(1)[1].eq(ground_truth.view(-1)).sum().item()

                test_acc /= len(fold_data[-1].dataset)
                test_accs.append(test_acc)

            if epoch % 10 == 0:
                print('fold:{}, epoch:{}, valid_acc:{:.4f}, test_acc:{:.4f}'.format(fold+1,epoch,valid_acc,test_acc))

    valid_accs = torch.tensor(valid_accs).view(folds, args.epochs_test)
    test_accs = torch.tensor(test_accs).view(folds, args.epochs_test)

    # max_valid_acc
    valid_accs, argmax = valid_accs.max(dim=-1)
    valid_acc_mean = round(valid_accs.mean().item()*100, 2)
    test_accs_argmax = test_accs[torch.arange(folds, dtype=torch.long), argmax] * 100
    test_acc_mean = round(test_accs_argmax.mean().item(), 2)
    test_acc_std = round(test_accs_argmax.std().item(), 2)
    print('test_accs:', test_accs_argmax)

    return valid_acc_mean, test_acc_mean, test_acc_std
