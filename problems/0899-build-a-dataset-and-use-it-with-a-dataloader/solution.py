import torch
from torch.utils.data import Dataset

class TensorPairDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        # TODO: store X and y on the instance
        self.X = X
        self.Y= y 


    def __len__(self):
       return len(self.X)

    def __getitem__(self, idx):
        return  self.X[idx],self.Y[idx]
