
import torch
import torch.nn.functional as F

def conv2d_via_unfold(x, W, b, stride=1, padding=0):

    # Get input dimensions
    N, C_in, H, width = x.shape

    # Get kernel dimensions
    C_out, _, kH, kW = W.shape

    # Extract sliding windows
    patches = F.unfold(
        x,
        kernel_size=(kH, kW),
        stride=stride,
        padding=padding
    )

    # Flatten the filters
    weight = W.reshape(C_out, -1)

    # Matrix multiplication
    output = torch.matmul(weight, patches)

    # Calculate output height and width
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (width + 2 * padding - kW) // stride + 1

    # Reshape into image format
    output = output.reshape(N, C_out, H_out, W_out)

    # Add bias
    output = output + b.view(1, C_out, 1, 1)

    return output
