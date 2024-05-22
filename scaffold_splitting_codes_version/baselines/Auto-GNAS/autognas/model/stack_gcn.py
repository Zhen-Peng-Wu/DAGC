import torch
from autognas.model.stack_gcn_encoder.gcn_encoder import GcnEncoder
from autognas.model.logger import gnn_architecture_performance_save,\
                                  test_performance_save,\
                                  model_save
from autognas.dynamic_configuration import optimizer_getter,  \
                                           loss_getter, \
                                           evaluator_getter, \
                                           downstream_task_model_getter
from dataset import load_k_fold

from dataset_utils.utils import auc_metric
from dataset_utils.chem_splitters import random_scaffold_split
import pandas as pd
import os
from torch_geometric.data import DataLoader
import numpy as np

class StackGcn(object):
    """
    Realizing stack GCN  model initializing, downstream task model initializing,
    model training validating and testing based on graph data and stack gcn architecture.

    Args:
        graph_data: graph data obj
            the target graph data object including required attributes:
            1.batch_train_x_list
            2.batch_train_edge_index_list
            3.batch_train_y_list
            4.batch_train_x_index_list
            5.batch_val_x_list
            6.batch_val_edge_index_list
            7.batch_val_y_list
            8.batch_val_x_index_list
            9.batch_test_x_list
            10.batch_test_edge_index_list
            11.batch_test_y_list
            12.batch_test_x_index_list
            13. num_features
            14.num_labels
            15.data_name
        gnn_architecture: list
            the stack gcn architecture describe
            for example: ['gcn', 'sum',  1, 64, 'tanh', 'gcn', 'sum', 1, 64, 'tanh']
        train_batch_size: int
            the batch size of train dataset
        val_batch_size: int
            the batch size of validation dataset
        test_batch_size: int
            the batch size of test dataset
        gnn_drop_out: float
            the drop out rate for stack gcn model for every layer
        train_epoch: int
            the model train epoch for validation
        train_epoch_test: int
            the model train epoch for testing
        bias: bool
            controlling whether add bias to the GNN model
        early_stop: bool
            controlling  whether use early stop mechanism in the model training process
        early_stop_patience: int
            controlling validation loss comparing cycle of the early stop mechanism
        opt_type: str
            the optimization function type for the model
        opt_parameter_dict: dict
            the hyper-parameter of selected optimizer
        loss_type: str
            the loss function type for the model
        val_evaluator_type: str
            the validation evaluating metric in the model training process
        test_evaluator_type: list
            the testing evaluating metric in the model testing process

    Returns:
        val_performance: float
            the validation result
    """

    def __init__(self,
                 graph_data,
                 downstream_task_type="node_classification",
                 gnn_architecture=['gcn', 'sum',  1, 128, 'relu', 'gcn', 'sum', 1, 64, 'linear'],
                 gnn_drop_out=0.6,
                 train_epoch=100,
                 train_epoch_test=100,
                 bias=True,
                 early_stop=False,
                 early_stop_patience=10,
                 opt_type="adam",
                 opt_parameter_dict={"learning_rate": 0.005, "l2_regularization_strength": 0.0005},
                 loss_type="nll_loss",
                 val_evaluator_type="accuracy",
                 test_evaluator_type=["accuracy", "precision", "recall", "f1_value"],
                 data_name='bace'
                 ):

        self.graph_data = graph_data
        self.downstream_task_type = downstream_task_type
        self.gnn_architecture = gnn_architecture
        self.gnn_drop_out = gnn_drop_out
        self.train_epoch = train_epoch
        self.train_epoch_test = train_epoch_test
        self.bias = bias
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.opt_type = opt_type
        self.opt_parameter_dict = opt_parameter_dict
        self.loss_type = loss_type
        self.val_evaluator_type = val_evaluator_type
        self.test_evaluator_type = test_evaluator_type
        self.data_name = data_name
        self.train_batch_id = 0
        self.val_batch_id = 0
        self.test_batch_id = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.val_evaluator = evaluator_getter(self.val_evaluator_type)

    def fit(self):
        # training on the training dataset based on training batch dataset

        self.gnn_model = GcnEncoder(self.gnn_architecture,
                                    self.graph_data[0].num_features,
                                    dropout=self.gnn_drop_out,
                                    bias=self.bias).to(self.device)

        self.downstream_task_model = downstream_task_model_getter(self.downstream_task_type,
                                                                  int(self.gnn_architecture[-2]),
                                                                  self.graph_data).to(self.device)
        self.optimizer = optimizer_getter(self.opt_type,
                                          self.gnn_model,
                                          self.downstream_task_model,
                                          self.opt_parameter_dict)


        num_tasks = int(self.graph_data[0].data.num_tasks[0])
        criterion = [torch.nn.CrossEntropyLoss() for i in range(num_tasks)]


        for epoch in range(1, self.train_epoch + 1):
            self.train_batch_id = 0
            self.gnn_model.train()
            self.downstream_task_model.train()
            y_true_train = []
            y_scores_train = []
            one_epoch_train_loss_list = []
            
            for train_data in self.graph_data[-3]:
                train_data = train_data.to(self.device)
                train_x = train_data.x
                train_edge_index = train_data.edge_index
                train_y = train_data.y
                train_batch = train_data.batch


                node_embedding_matrix = self.gnn_model(train_x, train_edge_index)
                train_predict_y = self.downstream_task_model(node_embedding_matrix,
                                                             train_batch,
                                                             mode="train")


                y = train_data.y.view(-1, num_tasks)
                y_true_train.append(y)
                y_scores_train.append(train_predict_y)

                self.train_batch_id += 1

                train_loss = 0.0
                for i in range(num_tasks):
                    is_valid = y[:, i] ** 2 > 0
                    train_loss += criterion[i](train_predict_y[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                one_epoch_train_loss_list.append(train_loss.item())

            train_loss = sum(one_epoch_train_loss_list)

            # validating on the validation dataset based on validation batch dataset in the training process
            self.gnn_model.eval()
            self.downstream_task_model.eval()
            self.val_batch_id = 0
            batch_val_loss_list = []
            y_true_val = []
            y_scores_val = []
            val_loss_list = []
            early_stop_flag = False


            for val_data in self.graph_data[-2]:
                val_data = val_data.to(self.device)
                val_x = val_data.x
                val_edge_index = val_data.edge_index
                val_y = val_data.y
                val_batch = val_data.batch

                node_embedding_matrix = self.gnn_model(val_x,
                                                       val_edge_index)

                val_predict_y = self.downstream_task_model(node_embedding_matrix,
                                                           val_batch,
                                                           mode="val")
                y = val_data.y.view(-1, num_tasks)
                y_true_val.append(y)
                y_scores_val.append(val_predict_y)


                val_loss = 0.0
                for i in range(num_tasks):
                    is_valid = y[:, i] ** 2 > 0
                    val_loss += criterion[i](val_predict_y[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())

                self.val_batch_id += 1
                batch_val_loss_list.append(val_loss.item())

            val_loss = sum(batch_val_loss_list)
            val_loss_list.append(val_loss)

            torch.cuda.empty_cache()

            if self.early_stop:
                early_stop_flag = self.early_stopping(val_loss_list,
                                                      self.early_stop_patience)

            if early_stop_flag:
                print("early stopping epoch:", epoch, "\n")
                break

        self.gnn_model.eval()
        self.downstream_task_model.eval()
        train_performance = auc_metric(y_true_train, y_scores_train)
        val_performance = auc_metric(y_true_val, y_scores_val)

        print("\n" + "stack gcn architecture:\t" + str(self.gnn_architecture) + "\n" +
              "train loss:\t" + str(train_loss) + "\n" +
              "train " + str(self.val_evaluator_type) + ":" + "\t" + str(train_performance) + "\n"
              "val loss:\t" + str(val_loss) + "\n" +
              "val " + str(self.val_evaluator_type) + ":" + "\t" + str(val_performance))

        gnn_architecture_performance_save(self.gnn_architecture, val_performance, self.data_name)

        return val_performance

    def evaluate(self, model_num=0):

        # train the model from the scratch based on strain_epoch_test
        print(25*"#", "testing, train from the scratch based on train_epoch_test", 25*"#")
        print("test gcn architecture "+str(model_num+1)+":\t" + str(self.gnn_architecture) + "\n")

        test_repeat = 20
        tests = []

        for time in range(test_repeat):
            batch_size = int(self.graph_data[0].batch_size)
            num_tasks = int(self.graph_data[0].num_tasks)

            smiles_list = pd.read_csv(os.path.join(self.graph_data[0].data_root, self.graph_data[0].data_name, 'processed/smiles.csv'), header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(self.graph_data[0], smiles_list, null_value=0,
                                                                               frac_train=0.8, frac_valid=0.1, frac_test=0.1)

            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
            time_data = [self.graph_data[0], None, None, None, train_loader, valid_loader, test_loader]

            criterion = [torch.nn.CrossEntropyLoss() for i in range(num_tasks)]


            print('###time {}, train/val/test:{},{},{}'.format(time + 1, len(time_data[-3].dataset),
                                                               len(time_data[-2].dataset), len(time_data[-1].dataset)))

            self.gnn_model = GcnEncoder(self.gnn_architecture,
                                        self.graph_data[0].num_features,
                                        dropout=self.gnn_drop_out,
                                        bias=self.bias).to(self.device)

            self.downstream_task_model = downstream_task_model_getter(self.downstream_task_type,
                                                                      int(self.gnn_architecture[-2]),
                                                                      self.graph_data).to(self.device)
            self.optimizer = optimizer_getter(self.opt_type,
                                              self.gnn_model,
                                              self.downstream_task_model,
                                              self.opt_parameter_dict)


            best_valid = 0.0
            best_valid_test = 0.0

            for epoch in range(1, self.train_epoch_test + 1):
                self.train_batch_id = 0
                self.gnn_model.train()
                self.downstream_task_model.train()
                y_true_train = []
                y_scores_train = []
                one_epoch_train_loss_list = []

                for train_data in time_data[-3]:
                    train_data = train_data.to(self.device)
                    train_x = train_data.x
                    train_edge_index = train_data.edge_index
                    train_y = train_data.y
                    train_batch = train_data.batch

                    node_embedding_matrix = self.gnn_model(train_x, train_edge_index)
                    train_predict_y = self.downstream_task_model(node_embedding_matrix,
                                                                 train_batch,
                                                                 mode="train")

                    self.train_batch_id += 1

                    y = train_data.y.view(-1, num_tasks)
                    y_true_train.append(y)
                    y_scores_train.append(train_predict_y)

                    train_loss = 0.0
                    for i in range(num_tasks):
                        is_valid = y[:, i] ** 2 > 0
                        train_loss += criterion[i](train_predict_y[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())

                    self.optimizer.zero_grad()
                    train_loss.backward()
                    self.optimizer.step()

                    one_epoch_train_loss_list.append(train_loss.item())


                # validating on the validation dataset based on validation batch dataset in the training process
                self.gnn_model.eval()
                self.downstream_task_model.eval()
                self.val_batch_id = 0
                batch_val_loss_list = []
                y_true_val = []
                y_scores_val = []

                for val_data in time_data[-2]:
                    val_data = val_data.to(self.device)
                    val_x = val_data.x
                    val_edge_index = val_data.edge_index
                    val_y = val_data.y
                    val_batch = val_data.batch

                    node_embedding_matrix = self.gnn_model(val_x,
                                                           val_edge_index)

                    val_predict_y = self.downstream_task_model(node_embedding_matrix,
                                                               val_batch,
                                                               mode="val")

                    y = val_data.y.view(-1, num_tasks)
                    y_true_val.append(y)
                    y_scores_val.append(val_predict_y)

                    val_loss = 0.0
                    for i in range(num_tasks):
                        is_valid = y[:, i] ** 2 > 0
                        val_loss += criterion[i](val_predict_y[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())


                    self.val_batch_id += 1
                    batch_val_loss_list.append(val_loss.item())
                valid_auc = auc_metric(y_true_val, y_scores_val)


                self.test_batch_id = 0
                y_true_test = []
                y_scores_test = []

                for test_data in time_data[-1]:
                    test_data = test_data.to(self.device)
                    test_x = test_data.x
                    test_edge_index = test_data.edge_index
                    test_y = test_data.y
                    test_batch = test_data.batch

                    node_embedding_matrix = self.gnn_model(test_x, test_edge_index)
                    test_predict_y = self.downstream_task_model(node_embedding_matrix,
                                                                test_batch,
                                                                mode="test")
                    y = test_data.y.view(-1, num_tasks)
                    y_true_test.append(y)
                    y_scores_test.append(test_predict_y)



                    self.test_batch_id += 1

                test_auc = auc_metric(y_true_test, y_scores_test)

                if valid_auc >= best_valid:
                    best_valid = valid_auc
                    best_valid_test = test_auc

                if epoch % 10 == 0:
                    print('time:{}, epoch:{}, valid_acc:{:.4f}, test_acc:{:.4f}'.format(time+1,epoch,valid_auc, test_auc))
            tests.append(best_valid_test)
            print("{}-th time in all {} times || Test_auc:{:.4f}".format(time + 1, test_repeat, test_auc))
        tests = np.array(tests)
        print("Final results --- Test_auc:{:.4f}±{:.4f}".format(tests.mean(), tests.std()))


    def early_stopping(self, val_loss_list, stop_patience):

        if len(val_loss_list) < stop_patience:
            return False

        if val_loss_list[-stop_patience:][0] > val_loss_list[-1]:
            return True

        else:
            return False
