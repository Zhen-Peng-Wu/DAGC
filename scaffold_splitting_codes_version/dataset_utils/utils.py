import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def auc_metric(y_true, y_scores):

    y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().detach().numpy()
    y_scores = np.nan_to_num(y_scores, nan=0.0)

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, 1, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    auc = sum(roc_list) / len(roc_list)
    return auc

