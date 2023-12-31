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
                 data_name='DD'
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


        self.loss = loss_getter(self.loss_type)

        for epoch in range(1, self.train_epoch + 1):
            self.train_batch_id = 0
            self.gnn_model.train()
            self.downstream_task_model.train()
            train_predict_y_list = []
            train_y_list = []
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

                self.train_batch_id += 1
                train_loss = self.loss.function(train_predict_y, train_y)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                one_epoch_train_loss_list.append(train_loss.item())
                train_predict_y_list.append(train_predict_y)
                train_y_list.append(train_y)

            train_loss = sum(one_epoch_train_loss_list)

            # validating on the validation dataset based on validation batch dataset in the training process
            self.gnn_model.eval()
            self.downstream_task_model.eval()
            self.val_batch_id = 0
            batch_val_loss_list = []
            val_predict_y_list = []
            val_y_list = []
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

                self.val_batch_id += 1
                val_loss = self.loss.function(val_predict_y, val_y)
                batch_val_loss_list.append(val_loss.item())
                val_predict_y_list.append(val_predict_y)
                val_y_list.append(val_y)

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
        train_performance = 0
        val_performance = 0
        for train_predict_y, train_y in zip(train_predict_y_list,
                                            train_y_list):

            train_performance += self.val_evaluator.function(train_predict_y,
                                                             train_y)

        for val_predict_y, val_y in zip(val_predict_y_list,
                                        val_y_list):

            val_performance += self.val_evaluator.function(val_predict_y,
                                                           val_y)

        train_performance = train_performance/self.train_batch_id

        val_performance = val_performance/self.val_batch_id

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

        folds = 20
        k_folds_data = load_k_fold(self.graph_data[0], folds, self.graph_data[-3].batch_size)

        valid_accs = []
        test_accs = []

        for fold, fold_data in enumerate(k_folds_data):
            print('###fold {}, train/val/test:{},{},{}'.format(fold + 1, len(fold_data[-3].dataset),
                                                               len(fold_data[-2].dataset), len(fold_data[-1].dataset)))

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

            self.loss = loss_getter(self.loss_type)

            for epoch in range(1, self.train_epoch_test + 1):
                self.train_batch_id = 0
                self.gnn_model.train()
                self.downstream_task_model.train()
                train_predict_y_list = []
                train_y_list = []
                one_epoch_train_loss_list = []

                for train_data in fold_data[-3]:
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
                    train_loss = self.loss.function(train_predict_y, train_y)
                    self.optimizer.zero_grad()
                    train_loss.backward()
                    self.optimizer.step()

                    one_epoch_train_loss_list.append(train_loss.item())
                    train_predict_y_list.append(train_predict_y)
                    train_y_list.append(train_y)


                # validating on the validation dataset based on validation batch dataset in the training process
                self.gnn_model.eval()
                self.downstream_task_model.eval()
                self.val_batch_id = 0
                batch_val_loss_list = []
                val_predict_y_list = []
                val_y_list = []

                for val_data in fold_data[-2]:
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

                    self.val_batch_id += 1
                    val_loss = self.loss.function(val_predict_y, val_y)
                    batch_val_loss_list.append(val_loss.item())
                    val_predict_y_list.append(val_predict_y)
                    val_y_list.append(val_y)
                val_performance = 0

                for val_predict_y, val_y in zip(val_predict_y_list, val_y_list):
                    val_performance += self.val_evaluator.function(val_predict_y, val_y)
                val_performance = val_performance / self.val_batch_id
                valid_accs.append(val_performance)

                self.test_batch_id = 0
                test_predict_y_list = []
                test_y_list = []

                for test_data in fold_data[-1]:
                    test_data = test_data.to(self.device)
                    test_x = test_data.x
                    test_edge_index = test_data.edge_index
                    test_y = test_data.y
                    test_batch = test_data.batch

                    node_embedding_matrix = self.gnn_model(test_x, test_edge_index)
                    test_predict_y = self.downstream_task_model(node_embedding_matrix,
                                                                test_batch,
                                                                mode="test")
                    self.test_batch_id += 1
                    test_predict_y_list.append(test_predict_y)
                    test_y_list.append(test_y)

                test_performance_all = []
                ### test_evaluator_type = ["accuracy", "precision", "recall", "f1_value"]
                for evaluator_type in self.test_evaluator_type:
                    test_performance = 0
                    for test_predict_y, test_y in zip(test_predict_y_list, test_y_list):
                        test_evaluator = evaluator_getter(evaluator_type)
                        test_performance = test_performance + test_evaluator.function(test_predict_y, test_y)
                    test_performance = test_performance / self.test_batch_id
                    test_performance_all.append(test_performance)
                test_accs.append(test_performance_all[0])

                if epoch % 10 == 0:
                    print('fold:{}, epoch:{}, valid_acc:{:.4f}, test_acc:{:.4f}'.format(fold+1,epoch,val_performance, test_performance_all[0]))

        valid_accs = torch.tensor(valid_accs).view(folds, self.train_epoch_test)
        test_accs = torch.tensor(test_accs).view(folds, self.train_epoch_test)

        # max_valid_acc
        valid_accs, argmax = valid_accs.max(dim=-1)
        test_accs_argmax = test_accs[torch.arange(folds, dtype=torch.long), argmax] * 100
        test_acc_mean = round(test_accs_argmax.mean().item(), 2)
        test_acc_std = round(test_accs_argmax.std().item(), 2)
        print("Test_acc:{}±{}".format(test_acc_mean, test_acc_std))

    def early_stopping(self, val_loss_list, stop_patience):

        if len(val_loss_list) < stop_patience:
            return False

        if val_loss_list[-stop_patience:][0] > val_loss_list[-1]:
            return True

        else:
            return False
