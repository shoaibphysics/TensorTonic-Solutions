import numpy as np

def projection_matrix(A: list) -> np.ndarray:
    """
    Returns the float64 projector onto the column space of A.
    """
    a = np.array(A, dtype=np.float64)
    b = a.T @ a
    b_inv = np.linalg.inv(b)
    return a@ b_inv @ a.T