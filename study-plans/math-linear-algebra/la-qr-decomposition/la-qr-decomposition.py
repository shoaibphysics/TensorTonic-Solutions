import numpy as np

def qr_decompose(A: list) -> dict:
    """
    Returns a dictionary containing float64 arrays Q and R.
    """
    matrix = np.asarray(A, dtype=np.float64)
    m, n = matrix.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        residual = matrix[:, k].copy()
        for j in range(k):
            R[j, k] = np.dot(Q[:, j], residual)
            residual -= R[j, k] * Q[:, j]
        R[k, k] = np.linalg.norm(residual)
        Q[:, k] = residual / R[k, k]
    return {"Q": Q, "R": R}