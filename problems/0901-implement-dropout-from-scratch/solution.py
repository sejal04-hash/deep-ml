
import torch

def dropout(x, p, training=True):
    if not training or p == 0:
        return x

    mask = (torch.rand_like(x) > p).float()

    return (x * mask) / (1 - p)
