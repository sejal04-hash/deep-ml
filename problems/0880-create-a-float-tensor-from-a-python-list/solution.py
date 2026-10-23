import torch

def to_float_tensor(values):
    value = torch.tensor(values ,dtype =torch.float32  )
    return value 