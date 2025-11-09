import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def cross_entropy_loss(y_true, y_pred, device, class_weights=None):
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    return loss_fn(y_pred, y_true)

loss_functions = {
    "bce": cross_entropy_loss,
    "cross_entropy": cross_entropy_loss,
}

def get_loss_function(loss_name):
    if loss_name in loss_functions:
        return loss_functions[loss_name]
    else:
        raise ValueError(f"Unknown loss function: {loss_name}.")