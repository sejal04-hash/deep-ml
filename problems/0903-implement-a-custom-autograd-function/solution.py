
import torch

class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        # Save input for backward
        ctx.save_for_backward(x)

        # Apply ReLU
        return torch.clamp(x, min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input
        (x,) = ctx.saved_tensors

        # Gradient is 1 where x > 0, otherwise 0
        grad_input = grad_output * (x > 0).to(grad_output.dtype)

        return grad_input

