import os
import torch
import time

logger_path = os.path.split(os.path.realpath(__file__))[0][:-(7+len('core'))] + "/logger"

def gnn_architecture_performance_save(gnn_architecture, performance, data_name):

    if not os.path.exists(logger_path+ '/' + str(data_name)):
        os.makedirs(logger_path+ '/' + str(data_name))

    with open(logger_path + '/' + str(data_name) + "/gnn_logger_" + str(data_name) + ".txt", "a+") as f:
        f.write(str(gnn_architecture) + ":" + str(performance) + "\n")

    print("gnn architecture and validation performance save")
    print("save path: ", logger_path + '/' + str(data_name)+ "/gnn_logger_" + str(data_name) + ".txt")
    print(50 * "=")


def model_save(model, optimizer, data_name, model_num, fold):
    torch.save(model, logger_path+ '/' + str(data_name)+"/model_" + data_name + "_num" + str(model_num)+'_'+str(fold) + ".pkl")

def model_load(data_name, model_num, fold):
    model = torch.load(logger_path + '/' + str(data_name) + "/model_" + data_name + "_num" + str(model_num) + '_' + str(fold) + ".pkl")
    return model
