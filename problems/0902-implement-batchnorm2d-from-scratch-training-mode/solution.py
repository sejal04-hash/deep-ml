
import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # Calculate channel-wise mean
    mean = x.mean(dim=(0, 2, 3), keepdim=True)

    # Calculate channel-wise population variance
    var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

    # Normalize input
    x_hat = (x - mean) / torch.sqrt(var + eps)

    # Reshape gamma and beta for broadcasting
    gamma = gamma.view(1, -1, 1, 1)
    beta = beta.view(1, -1, 1, 1)

    # Scale and shift
    y = gamma * x_hat + beta

    return y
