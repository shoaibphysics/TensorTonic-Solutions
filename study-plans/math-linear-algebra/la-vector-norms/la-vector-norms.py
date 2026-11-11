import numpy as np

def vector_norms(v: list) -> np.ndarray:
    """
    Returns a float64 array containing the L1, L2, and infinity norms.
    """
    arr = np.array(v, dtype=np.float64)
    L1 = np.sum(np.abs(arr))
    L2 = np.linalg.norm(arr)
    Linf = np.max(np.abs(arr))
    return np.stack([L1,L2,Linf])