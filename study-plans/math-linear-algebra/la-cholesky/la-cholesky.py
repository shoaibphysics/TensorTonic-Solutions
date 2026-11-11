import numpy as np

def cholesky_decompose(A: list):
    """
    Returns a float64 lower-triangular array, or None.
    """
    a = np.array(A, dtype=np.float64)
    L = np.zeros_like(a)
    c = L.shape[1]
    for j in range(c):
        val = a[j,j]-np.sum(L[j,:j]**2)
        if val <= 0:
            return None
        L[j,j] = np.sqrt(val)
        L[j+1:,j] = (1/L[j,j])*(a[j+1:,j]-L[j+1:,:j]@L.T[:j,j])
    return L