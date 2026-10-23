import torch

def add_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    add = x + b 

    return add 
