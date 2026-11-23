import torch

def grouped_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns an attention tensor with the same shape as hidden_states.
    """
    B, T, D = hidden_states.shape
    head_dim = D // num_query_heads
    g = num_query_heads // num_kv_heads

    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    
    assert D == w_q.shape[0] , "dimension of w_q not compatible"
    assert head_dim*num_kv_heads == w_v.shape[-1] , "dimension of w_q, w_v not compatible"

    q = (hidden_states @ w_q).reshape(B,T,num_query_heads, head_dim).transpose(1,2) # (B, hq ,T,d)
    k = (hidden_states @ w_k).reshape(B,T,num_kv_heads, head_dim).transpose(1,2) # (B, hkv, T d)
    v = (hidden_states @ w_v).reshape(B,T,num_kv_heads, head_dim).transpose(1,2) # (B, hkv, T d)

    k = k.repeat_interleave(g, dim = 1)
    v = v.repeat_interleave(g, dim = 1)

    scores = (q @ k.transpose(-2,-1))/(head_dim**(0.5))

    if causal:
        mask = torch.ones((T,T), dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = scores.softmax(dim=-1)
    values = (weights @ v).transpose(1,2).contiguous().view(B,T,D)
    return values @ w_o
    
