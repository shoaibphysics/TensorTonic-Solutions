import torch
import torch.nn as nn

def train_with_early_stopping(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, max_epochs: int, patience: int) -> dict:
    """
    Returns train_losses and val_losses as float lists, and stopped_epoch as an int.
    """
    count = 0
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, max_epochs+1):
        model.train()
        train_loss = 0.0
        for x,y in train_loader:
            loss = criterion(model(x),y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                loss = criterion(model(x),y)
                val_loss += loss.item()
            val_losses.append(val_loss / len(val_loader))

        # early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            count = 0
        else:
            count+=1
            if count >= patience:
                return {"train_losses": train_losses, 
                        "val_losses": val_losses, 
                        "stopped_epoch": epoch}
    return {"train_losses": train_losses, 
            "val_losses": val_losses, 
            "stopped_epoch": epoch}
            
            




        
    
