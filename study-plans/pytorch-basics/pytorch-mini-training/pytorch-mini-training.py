import torch
import torch.nn as nn

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Returns the mean batch loss as a Python float.
    """
    total_loss = 0
    model.train()
    for x, y in dataloader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss.item()/len(dataloader)
        
