import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    if method == "relu":
        return torch.relu(x)
    if method == "sigmoid":
        return torch.sigmoid(x)
    if method == "tanh":
        return torch.tanh(x)
    return torch.where(x > 0, x, 0.01 * x)
