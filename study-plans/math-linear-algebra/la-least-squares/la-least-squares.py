import numpy as np

def least_squares(A: list, b: list) -> np.ndarray:
    """
    Returns the float64 least-squares solution vector.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    Q,R = np.linalg.qr(A, mode="reduced")
    return np.linalg.solve(R,Q.T@b)