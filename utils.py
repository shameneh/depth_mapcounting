import numpy as np
import torch


def denormalize(tensor):
    mean = [0.5, 0.5, 0.5]
    std = [0.225,0.225,0.225]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

class weighted_mse_loss():
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, inputs, targets,weights=None):
        loss = (inputs-targets)**2
        if weights is not None:
            loss *=weights.expand_as(loss)
        loss=torch.mean(loss)
        return loss
     

