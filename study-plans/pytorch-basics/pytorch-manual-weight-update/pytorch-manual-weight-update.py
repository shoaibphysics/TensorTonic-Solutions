import torch
import torch.nn as nn

def manual_train_step(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module, lr: float) -> float:
    """
    Returns the pre-update batch loss as a Python float.
    """
    output = model(X)
    loss = criterion(output,y)
    loss.backward()
    with torch.no_grad():
        for params in model.parameters():
            params -= lr*params.grad
            params.grad.zero_()
    
        
    return loss.item()
