import os
import os.path as osp
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from model_search import Network
from dataset import load_data
from logging_util import init_logger

from dataset_utils.chem_datasets import MoleculeDataset
from dataset_utils.chem_pre import pre_filter, pre_transform
from dataset_utils.chem_splitters import random_scaffold_split
from dataset_utils.utils import auc_metric
import pandas as pd
from torch_geometric.data import DataLoader

graph_classification_dataset = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'hiv', 'muv']
node_classification_dataset = ['Cora','CiteSeer', 'PubMed','Amazon_Computers','Coauthor_CS','Coauthor_Physics','Amazon_Photo',
                               'small_Reddit', 'small_arxiv','Reddit','ogbn-arxiv']


## the example of run command: python train_search.py  --data clintox --gpu 0
parser = argparse.ArgumentParser("pas-train-search")
parser.add_argument('--data', type=str, default='clintox', help='location of the data corpus')
parser.add_argument('--record_time', action='store_true', default=False, help='used for run_with_record_time func')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0005, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--epsilon', type=float, default=0.0, help='the explore rate in the gradient descent process')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=0.08, help='learning rate for arch encoding')
parser.add_argument('--arch_learning_rate_min', type=float, default=0.005, help='minimum learning rate for arch encoding')
# parser.add_argument('--cos_arch_lr', action='store_true', default=False, help='lr decay for learning rate')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--with_conv_linear', type=bool, default=False, help=' in NAMixOp with linear op')
parser.add_argument('--num_layers', type=int, default=2, help='num of layers of GNN method.')
parser.add_argument('--withoutjk', action='store_true', default=False, help='remove la aggregtor')
parser.add_argument('--alpha_mode', type=str, default='train_loss', help='how to update alpha', choices=['train_loss', 'valid_loss', 'valid_acc'])
parser.add_argument('--search_act', action='store_true', default=False, help='search act in supernet.')
parser.add_argument('--hidden_size',  type=int, default=64, help='default hidden_size in supernet')
parser.add_argument('--BN',  type=int, default=64, help='default hidden_size in supernet')
parser.add_argument('--num_sampled_archs',  type=int, default=5, help='sample archs from supernet')


###for ablation stuty
parser.add_argument('--remove_pooling', action='store_true', default=False, help='remove pooling block.')
parser.add_argument('--remove_readout', action='store_true', default=False, help='exp5, only search the last readout block.')
parser.add_argument('--remove_jk', action='store_true', default=False, help='remove ensemble block, Graph representation = Z3')

#in the stage of update theta.
parser.add_argument('--temp', type=float, default=0.2, help=' temperature in gumble softmax.')
parser.add_argument('--loc_mean', type=float, default=10.0, help='initial mean value to generate the location')
parser.add_argument('--loc_std', type=float, default=0.01, help='initial std to generate the location')
parser.add_argument('--lamda', type=int, default=2, help='sample lamda architectures in calculate natural policy gradient.')
parser.add_argument('--adapt_delta', action='store_true', default=False, help='adaptive delta in update theta.')
parser.add_argument('--delta', type=float, default=1.0, help='a fixed delta in update theta.')
parser.add_argument('--w_update_epoch', type=int, default=1, help='epoches in update W')
parser.add_argument('--model_type', type=str, default='snas', help='how to update alpha', choices=['darts', 'snas'])

args = parser.parse_args()
args.graph_classification_dataset = graph_classification_dataset
args.node_classification_dataset = node_classification_dataset
torch.set_printoptions(precision=4)
def main():
    global device
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    args.save = 'logs/search-{}'.format(args.save)
    if not os.path.exists(args.save):
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_filename = os.path.join(args.save, 'log.txt')
    init_logger('', log_filename, logging.INFO, False)
    print('*************log_filename=%s************' % log_filename)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args.__dict__)

    if args.data in args.graph_classification_dataset:
        args.data_root = '../../chem_data'
        graph_data = MoleculeDataset(os.path.join(args.data_root, args.data), dataset=args.data,
                                     pre_filter=pre_filter, pre_transform=pre_transform)
        args.num_features = graph_data.num_features
        args.num_classes = graph_data.num_classes
        args.num_tasks = int(graph_data.data.num_tasks[0])

    hidden_size = args.hidden_size

    smiles_list = pd.read_csv(os.path.join(args.data_root, args.data, 'processed/smiles.csv'), header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = random_scaffold_split(graph_data, smiles_list, null_value=0,
                                                                       frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    data = [graph_data, None, None, None, train_loader, valid_loader, test_loader]

    criterion = [torch.nn.CrossEntropyLoss() for i in range(args.num_tasks)]


    num_nodes = graph_data.data.x.size(0)
    # criterion = criterion.cuda()

    model = Network(criterion, args.num_features, args.num_classes, hidden_size, epsilon=args.epsilon,
                    args=args, with_conv_linear=args.with_conv_linear, num_layers=args.num_layers, num_nodes=num_nodes)
    print('with_conv_linear: ', args.with_conv_linear)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


    optimizer = torch.optim.Adam(
        model.parameters(),
        args.learning_rate,
        # momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    arch_optimizer = torch.optim.Adam(
        model.arch_parameters(),
        lr=args.arch_learning_rate,
        # betas=(0.5, 0.999),
        weight_decay=args.arch_weight_decay) #fix lr in arch_optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_arch = torch.optim.lr_scheduler.CosineAnnealingLR(arch_optimizer, float(args.epochs), eta_min=args.arch_learning_rate_min)
    # scheduler_arch = torch.optim.lr_scheduler.ExponentialLR(arch_optimizer, 0.98)
    # architect = Architect(model, args) # arch_parameter.
    test_auc_with_time = []
    cur_t = 0
    for epoch in range(args.epochs):
        t1 = time.time()
        lr = scheduler.get_lr()[0]
        arch_lr = scheduler_arch.get_lr()[0]
        if epoch % 1 == 0:
            logging.info('epoch %d lr %e', epoch, lr)
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)

        train_auc, train_obj = train_graph(data, model, criterion, optimizer, arch_optimizer,lr,arch_lr)
        scheduler.step()

        # valid_auc, valid_obj = infer(data, model, criterion, test=False)
        # test_auc, test_obj = infer(data, model, criterion, test=True)
        valid_obj, valid_auc, test_obj, test_auc = infer_graph(data, model, criterion)
        s_valid_obj, s_valid_auc, s_test_obj, s_test_auc = infer_graph(data, model, criterion, mode='evaluate_single_path')
        scheduler_arch.step()

        if epoch % 1 == 0:
            logging.info('epoch=%s, train_auc=%f, train_loss=%f, valid_auc=%f, valid_loss=%f, test_auc=%f, test_loss=%f, explore_num=%s', epoch, train_auc, train_obj, valid_auc, valid_obj, test_auc, test_obj, model.explore_num)
            print('epoch={}, train_auc={:.04f}, train_loss={:.04f},valid_auc={:.04f}, valid_loss={:.04f},test_auc={:.04f},test_loss={:.04f}, explore_num={}'.
                  format(epoch, train_auc, train_obj, valid_auc, valid_obj, test_auc, test_obj, model.explore_num))

            logging.info('single path evaluate. epoch=%s, valid_auc=%f, valid_loss=%f, test_auc=%f, test_loss=%f, explore_num=%s', epoch, s_valid_auc, s_valid_obj, s_test_auc, s_test_obj, model.explore_num)
            print('single_path evaluation.  epoch={}, valid_auc={:.04f}, valid_loss={:.04f},test_auc={:.04f},test_loss={:.04f}, explore_num={}'.
                  format(epoch, s_valid_auc, s_valid_obj, s_test_auc, s_test_obj, model.explore_num))
        utils.save(model, os.path.join(args.save, 'weights.pt'))

    logging.shutdown()
    return genotype

def train_graph(data, model,  criterion, model_optimizer, arch_optimizer, lr,arch_lr):
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []

    # data:[dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
    train_iters = data[4].__len__()//args.w_update_epoch + 1
    print('train_iters:{},train_data_num:{}'.format(train_iters, range(train_iters * args.w_update_epoch)))
    from itertools import cycle
    zip_train_data = list(zip(range(train_iters * args.w_update_epoch), cycle(data[4])))
    zip_valid_data = list(zip(range(train_iters), cycle(data[5])))

    # for train_data, valid_data in zip_list:
    for iter in range(train_iters):
        for i in range(args.w_update_epoch):
            model_optimizer.zero_grad()
            arch_optimizer.zero_grad()

            train_data = zip_train_data[iter*args.w_update_epoch+i][1].to(device)

            output, sample_z = model(train_data)
            output = output.to(device)
            y = train_data.y.view(-1, args.num_tasks)
            y_true.append(y)
            y_pred.append(output)

            #error loss and resource loss
            if args.data =='COLORS-3':
                # error_loss = (output-train_data.y.long()).pow(2).mean()
                error_loss = criterion(output, train_data.y.long())

            else:
                error_loss = 0.0
                for i in range(args.num_tasks):
                    is_valid = y[:, i] ** 2 > 0
                    error_loss += criterion[i](output[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())
            print('loss:{:.08f}'.format(error_loss.item()))

            total_loss += error_loss.item()
            arch_optimizer.zero_grad()
            error_loss.backward(retain_graph=True)
            model_optimizer.step()
        model_optimizer.zero_grad()
        if args.alpha_mode =='train_loss':
            arch_optimizer.step()
        if args.alpha_mode =='valid_loss':
            valid_data = zip_valid_data[iter][1].to(device)
            model_optimizer.zero_grad()
            arch_optimizer.zero_grad()
            output, _ = model(valid_data)
            output = output.to(device)
            y = valid_data.y.view(-1, args.num_tasks)

            error_loss = 0.0
            for i in range(args.num_tasks):
                is_valid = y[:, i] ** 2 > 0
                error_loss += criterion[i](output[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())
            error_loss.backward()
            arch_optimizer.step()

    auc = auc_metric(y_true, y_pred)
    return auc, total_loss / len(data[4].dataset)

def infer_graph(data_, model, criterion, mode='none'):

    model.eval()
    valid_loss, test_loss = 0, 0
    y_true, y_pred = [], []
    y_true_test, y_pred_test = [], []

    for val_data in data_[5]:
        val_data = val_data.to(device)
        with torch.no_grad():
            logits, _ = model(val_data, mode=mode)
            logits = logits.to(device)
            y = val_data.y.view(-1, args.num_tasks)
            y_true.append(y)
            y_pred.append(logits)
        if args.data == 'COLORS-3':
            # loss = (logits-target.long()).pow(2).mean()
            loss = criterion(logits, target.long())
        else:
            loss = 0.0
            for i in range(args.num_tasks):
                is_valid = y[:, i] ** 2 > 0
                loss += criterion[i](logits[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())

        valid_loss += loss.item()
    for test_data in data_[6]:
        test_data = test_data.to(device)
        with torch.no_grad():
            logits,_ = model(test_data, mode=mode)
            logits = logits.to(device)
            y = test_data.y.view(-1, args.num_tasks)
            y_true_test.append(y)
            y_pred_test.append(logits)
        if args.data == 'COLORS-3':
            loss = criterion(logits, target.long())
            # loss = (logits-target.long()).pow(2).mean()
        else:
            loss = 0.0
            for i in range(args.num_tasks):
                is_valid = y[:, i] ** 2 > 0
                loss += criterion[i](logits[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())

        test_loss += loss.item()
    auc = auc_metric(y_true, y_pred)
    auc_test = auc_metric(y_true_test, y_pred_test)
    return valid_loss/len(data_[5].dataset), auc, \
           test_loss/len(data_[6].dataset), auc_test


def run_by_seed():
    res = []
    for i in range(args.num_sampled_archs):
        print('searched {}-th for {}...'.format(i+1, args.data))
        args.save = '{}-{}'.format(args.data, time.strftime("%Y%m%d-%H%M%S"))
        seed = np.random.randint(0, 10000)
        args.seed = seed
        genotype = main()
        res.append('seed={},genotype={},saved_dir={}'.format(seed, genotype, args.save))
    filename = './exp_res/%s-searched_res-%s-eps%s-reg%s.txt' % (args.data, time.strftime('%Y%m%d-%H%M%S'), args.epsilon, args.weight_decay)
    fw = open(filename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print('searched res for {} saved in {}'.format(args.data, filename))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Please specify the required config info!!!')
        sys.exit(0)
    run_by_seed()


