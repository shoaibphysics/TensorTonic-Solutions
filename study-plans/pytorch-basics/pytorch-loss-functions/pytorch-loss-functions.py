import torch

def compute_loss(pred: torch.Tensor, target: torch.Tensor, method: str, delta: float = 1.0) -> float:
    """
    Returns the mean loss as a Python float.
    """
    if method == "mse":
        return ((pred - target)**2).mean().item()
    elif method == "cross_entropy":
        shifted = pred - pred.max(dim=-1, keepdim=True).values # to make exp evaluation stable
        log_partition = shifted.exp().sum(dim=1).log()
        batch_index = torch.arange(pred.shape[0])
        target_index = target.to(dtype = torch.long)
        target_logits = shifted[batch_index, target_index]
        return (log_partition - target_logits).mean().item()
    elif method == "huber":
        a = torch.abs(pred - target)
        return torch.where(a>delta, delta*(a - 0.5*delta), 0.5 * a**2).mean().item()
        
