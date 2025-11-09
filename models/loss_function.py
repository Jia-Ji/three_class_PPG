import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter

def binary_cross_entropy_with_logits(y_true, y_pred, device):
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device))
    loss_fn = nn.CrossEntropyLoss()
    
    # counts = Counter({0: 41.45, 1: 58.55})
    # total = sum(counts.values())

    # weights = [total / counts[0], total / counts[1]] 
    # weights = torch.tensor(weights, dtype=torch.float32).to(device)
    # loss_fn = nn.CrossEntropyLoss(weight = weights)
    # loss = F.cross_entropy(y_pred, y_true)
    loss = loss_fn(y_pred, y_true)
    return loss

loss_functions = {"bce": binary_cross_entropy_with_logits}

def get_loss_function(loss_name):
    if loss_name in loss_functions:
        return loss_functions[loss_name]
    else:
        raise ValueError(f"Unknown loss function: {loss_name}.")