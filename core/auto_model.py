import os
from core.search_space.search_space_config import SearchSpace
from core.search_algorithm.graphnas_search_algorithm import Search
from core.estimation import Scratch_Train_Test

class AutoModel(object):
    """
    The top API to realize gnn architecture search and model testing automatically.

    Using search algorithm samples gnn architectures and evaluate
    corresponding performance,testing the top k model from the sampled
    gnn architectures based on performance.

    """

    def __init__(self, graph_data, args):

        self.graph_data = graph_data
        self.args = args
        self.search_space = SearchSpace()

        print("stack gcn architecture information:\t", self.search_space.stack_gcn_architecture)

        self.search_algorithm = Search(self.graph_data,
                                       self.args,
                                       self.search_space)

        self.search_model()


        self.derive_target_model()


    def search_model(self):

        self.search_algorithm.search_operator()

    def derive_target_model(self):

        path = os.path.split(os.path.realpath(__file__))[0][:-(len('core') + 1)] + "/logger" + '/' + str(self.args.data_name) + "/gnn_logger_"+ str(self.args.data_name)
        architecture_performance_list = self.gnn_architecture_performance_load(path)
        gnn_architecture_performance_dict = {}
        gnn_architecture_list = []
        performance_list = []

        for line in architecture_performance_list:
            line = line.split(":")
            gnn_architecture = eval(line[0])
            performance = eval(line[1].replace("\n", ""))
            gnn_architecture_list.append(gnn_architecture)
            performance_list.append(performance)

        for key, value in zip(gnn_architecture_list, performance_list):
            gnn_architecture_performance_dict[str(key)] = value

        ranked_gnn_architecture_performance_dict = sorted(gnn_architecture_performance_dict.items(),
                                                          key=lambda x: x[1],
                                                          reverse=True)

        sorted_gnn_architecture_list = []
        sorted_performance = []

        top_k = int(self.args.test_gnn_num)
        i = 0
        for key, value in ranked_gnn_architecture_performance_dict:
            if i == top_k:
                break
            else:
                sorted_gnn_architecture_list.append(eval(key))
                sorted_performance.append(value)
                i += 1

        print(35 * "=" + " the testing start " + 35 * "=")

        model_num = [num for num in range(len(sorted_gnn_architecture_list))]

        for target_architecture, num in zip(sorted_gnn_architecture_list, model_num):
            self.target_architecture = target_architecture

            print("test gnn architecture {}:\t".format(num + 1), str(self.target_architecture))

            ## train from scratch
            test_repeat = 5
            for i in range(test_repeat):
                valid_acc, test_acc, test_acc_std = Scratch_Train_Test(self.target_architecture, self.graph_data, args=self.args)

                print("{}-th run in all {} runs || Test_acc:{}±{}".format(i+1, test_repeat, test_acc, test_acc_std))

        print(35 * "=" + " the testing ending " + 35 * "=")

    def gnn_architecture_performance_load(self, path):

        with open(path + ".txt", "r") as f:
            gnn_architecture_performance_list = f.readlines()
        return gnn_architecture_performance_list
