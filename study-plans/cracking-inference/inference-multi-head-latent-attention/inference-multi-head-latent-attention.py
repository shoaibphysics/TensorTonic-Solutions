import torch

def multi_head_latent_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_down: torch.Tensor,
    w_up_k: torch.Tensor,
    w_up_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> tuple:
    """
    Returns (output, latent), tensors shaped (batch, seq, model width) and (batch, seq, latent width).
    """
    B,T,D = hidden_states.shape
    head_dim = D // num_heads
    if D % num_heads != 0:
        raise ValueError("head dimensions not compatible")
    q = (hidden_states @ w_q).reshape(B,T,num_heads,head_dim).transpose(1,2) # (B,h,T,d)
    c = (hidden_states  @ w_down) # (B,T,ld)

    k = (c @ w_up_k).reshape(B,T,num_heads,head_dim).transpose(1,2) # (B,h,T,d)
    v = (c @ w_up_v).reshape(B,T,num_heads,head_dim).transpose(1,2) # (B,h,T,d)

    scores = (q @ k.transpose(-2,-1))/(head_dim**(0.5)) # (B,h,T,d)

    if causal:
        mask = torch.ones((T,T), dtype= torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = scores.softmax(dim=-1)
    values = (weights @ v).transpose(1,2).contiguous().reshape(B,T,D)
    output = values @ w_o
    

    return output, c
