import torch

def multi_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns an attention tensor with the same shape as hidden_states.
    """
    B, T, D = hidden_states.shape
    _, d = w_k.shape
    assert num_query_heads*d == D , "Dimensions of heads are not compatible"

    q = (hidden_states @ w_q).reshape(B,T,num_query_heads, d).transpose(1,2) # (B,H,T,d)
    k = (hidden_states @ w_k)[:,None,:,:] # (B,1,T,d)
    v = (hidden_states @ w_v)[:,None,:,:] # (B,1,T,d)

    scores = (q @ k.transpose(-2,-1))/(d**(0.5)) # (B,H,T,T)
    if causal:
        mask = torch.ones((T,T), dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = scores.softmax(dim=-1)  # (B,H,T,T)
    values = (weights @ v).transpose(1,2).contiguous().reshape(B,T,D)
    return values @ w_o
    
    
    
