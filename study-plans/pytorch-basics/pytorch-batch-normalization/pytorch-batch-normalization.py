import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    mu = X.mean(dim=0, keepdim=True)
    variance = ((X-mu)**2).mean(dim = 0, keepdim=True)
    denominator = torch.rsqrt(variance+eps)
    return gamma*(X-mu)*denominator + beta
