import numpy as np

def matrix_rank(A: list) -> int:
    """
    Returns the numerical rank as a Python integer.
    """
    A = np.array(A, dtype=np.float64)
    s = np.linalg.svd(A, compute_uv = False)
    m = np.max(s)
    eps = np.finfo(np.float64).eps
    tau = max(A.shape)*eps*m
    mask = (s > tau).astype(np.int64)
    return int(np.sum(mask))
    
    