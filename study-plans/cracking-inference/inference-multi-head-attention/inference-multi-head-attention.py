import torch

def multi_head_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as hidden_states.
    """
    B ,T, d = hidden_states.shape
    dim_heads =  d // num_heads
    q = (hidden_states @ w_q).reshape(B,T, num_heads ,dim_heads).transpose(-3,-2) # (B,H,T,d)
    k = (hidden_states @ w_k).reshape(B,T, num_heads ,dim_heads).transpose(-3,-2)
    v = (hidden_states @ w_v).reshape(B,T, num_heads ,dim_heads).transpose(-3,-2)

    scores = (q @ k.transpose(-2,-1))/(dim_heads**(0.5))

    if causal:
        mask = torch.tril(torch.ones((T,T) , dtype = torch.bool) )
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = scores.softmax(dim=-1)
    output = (weights @ v).transpose(-3,-2).contiguous().reshape(B,T,d)  
    return output @ w_o
    


    
    
