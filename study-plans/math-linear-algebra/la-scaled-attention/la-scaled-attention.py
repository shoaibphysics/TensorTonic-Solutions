import numpy as np

def scaled_dot_product_attention(Q: list, K: list, V: list) -> np.ndarray:
    """
    Returns the scaled-attention output as a float64 array.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    d_k = Q.shape[-1]
    d_v = V.shape[-1]

    S = (Q @ K.T)/np.sqrt(d_k)
    S = S - np.max(S, axis = -1, keepdims=True)
    expS = np.exp(S)
    P = expS / np.sum(expS, axis=-1, keepdims=True)
    return P@V