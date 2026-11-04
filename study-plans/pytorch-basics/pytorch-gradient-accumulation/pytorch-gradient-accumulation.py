import torch

def gradient_accumulation(w_init: torch.Tensor, micro_batches: list, lr: float, accum_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (final_weights, last_mean_gradient) as float32 tensors.
    """
    w = w_init.detach().clone().requires_grad_(True)
    last_mean_gradient = None

    for i, (x, t) in enumerate(micro_batches):
        L = ((w.T @ x-t)**2)
        L.backward()
        w_grad = w.grad

        if (i+1)% accum_steps == 0:
            last_mean_gradient = w.grad.clone() / accum_steps
            with torch.no_grad():
                w -= lr*last_mean_gradient
            w.grad.zero_()
    return w, last_mean_gradient
            
        
