import torch

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    xr  = x.reshape(-1).reshape(new_shape)
    return  xr 

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    xy = x.transpose(-1,-2)

    return xy
