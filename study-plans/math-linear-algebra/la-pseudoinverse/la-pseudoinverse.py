import numpy as np

def pseudoinverse(A: list) -> np.ndarray:
    """
    Returns the Moore-Penrose pseudoinverse as a float64 array.
    """
    a = np.array(A, dtype = np.float64)
    U, s, Vt = np.linalg.svd(a, full_matrices = False)
    s_inv = np.zeros_like(s)
    mask = s > 1e-10
    s_inv[mask] = 1/s[mask]
    return Vt.T @ np.diag(s_inv) @ U.T