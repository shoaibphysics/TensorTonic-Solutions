import numpy as np

def low_rank_approximation(A: list, r: int) -> np.ndarray:
    """
    Returns the best rank-r float64 approximation.
    """
    a = np.asarray(A, dtype=np.float64)
    U, s, Vt = np.linalg.svd(a, full_matrices = False)
    return U[:, 0:r]@ np.diag(s[0:r])@Vt[0:r,:]