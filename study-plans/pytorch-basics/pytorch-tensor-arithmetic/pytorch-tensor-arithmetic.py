import torch

def tensor_op(x: torch.Tensor, y: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the operation result as a float32 tensor.
    """
    if op == "add":
        return x+y
    if op == "multiply":
        return x*y
    if op == "matmul":
        return x@y
    if op == "power":
        return x**y
    return torch.maximum(x,y)
        
