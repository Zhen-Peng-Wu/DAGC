import torch

class Optimizer:
    """
    Realizing the adam optimizer object

    Args:
       gnn_model: model object
            the pytorch model object
       optimizer_parameter_dict: dict
            the hyper parameter for optimizer

    Returns:
       optimizer: optimizier object
            the adam optimizer object
    """

    def __init__(self,
                 gnn_model,
                 downstream_task_model,
                 optimizer_parameter_dict):

        self.gnn_model = gnn_model
        self.downstream_task_model = downstream_task_model
        self.learning_rate = optimizer_parameter_dict["learning_rate"]
        self.l2_regularization_strength = optimizer_parameter_dict["l2_regularization_strength"]

    def function(self):
        optimizer = torch.optim.Adam([{'params': self.gnn_model.parameters()},
                                      {'params': self.downstream_task_model.parameters()}],
                                     lr=self.learning_rate,
                                     weight_decay=self.l2_regularization_strength)
        return optimizer

