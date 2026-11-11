import numpy as np

def norm_gate(X: list, W: list, threshold: float) -> np.ndarray:
    """
    Returns an (n, k) float64 matrix of norm-gated transformed rows.
    """
    x=np.array(X, dtype=np.float64)
    w=np.array(W, dtype=np.float64)
    Z = x@w
    norms = np.linalg.norm(Z,axis=-1)
    mask = norms >= threshold
    return Z*mask[:,None]
